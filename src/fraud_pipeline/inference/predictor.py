"""GATv2 replay inference using the offline feature/scaler/graph policy."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..features.causal import COMPLETED, REQUESTS, State, microseconds, names, replay
from ..features.dataset import graph_context
from ..features.export import NODE_TYPES
from ..ingestion.request_loader import file_hash
from ..registry.save_load import load
from .contracts import normalize_event, unwrap_record
from .response import build_response
from .account_catalog import ACCOUNT_IDENTITY_POLICY, request_account_catalog


class ReplayPredictor:
    account_identity_policy = ACCOUNT_IDENTITY_POLICY

    def __init__(self, bundle, *, device='cpu'):
        self.directory = Path(bundle)
        self.bundle = json.loads((self.directory/'bundle.json').read_text())
        expected_files = {'model.pt', 'manifest.json', 'preprocessing.json', 'node_preprocessing.json', 'nodes.json', 'selection.json'}
        if self.bundle['schema_version'] != 1 or set(self.bundle['files']) != expected_files:
            raise ValueError('Unsupported/incomplete inference bundle')
        for name, digest in self.bundle['files'].items():
            if file_hash(self.directory/name) != digest:
                raise ValueError(f'Bundle checksum mismatch: {name}')
        if self.bundle['feature_manifest_sha256'] != file_hash(self.directory/'manifest.json'):
            raise ValueError('Bundle feature identity mismatch')
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        self.options = self.manifest['options']
        if names(self.options) != self.manifest['feature_names']:
            raise ValueError('Serving feature schema differs from training')
        self.preprocessing = json.loads((self.directory/'preprocessing.json').read_text())
        self.node_preprocessing = json.loads((self.directory/'node_preprocessing.json').read_text())
        self.nodes = json.loads((self.directory/'nodes.json').read_text())
        self.mapping = {n['id']: i for i, n in enumerate(self.nodes)}
        self.zone = datetime.strptime(self.manifest['timezone'], '%z').tzinfo
        self.model, checkpoint = load(self.directory/'model.pt', device=device,
                                      expected_feature_sha256=self.bundle['feature_manifest_sha256'])
        if checkpoint['model_config'].get('kind', 'gatv2') != 'gatv2':
            raise ValueError('Only GATv2 inference is supported')
        selection = json.loads((self.directory/'selection.json').read_text())
        chosen = selection['models']['gatv2']
        if (selection['status'] != 'frozen_before_test' or selection['selected_model'] != 'gatv2'
                or selection['feature_manifest_sha256'] != self.bundle['feature_manifest_sha256']
                or chosen['sha256'] != self.bundle['files']['model.pt']
                or type(self.bundle['threshold']) not in (int, float)
                or not np.isfinite(self.bundle['threshold']) or not 0 <= self.bundle['threshold'] <= 1
                or chosen['threshold'] != self.bundle['threshold']
                or checkpoint['metadata']['snapshot_seconds'] != self.bundle['snapshot_seconds']):
            raise ValueError('Serving policy differs from frozen training/validation artifacts')
        for name in ('preprocessing.json', 'node_preprocessing.json', 'nodes.json'):
            if self.bundle['files'][name] != self.manifest['files'][name]:
                raise ValueError(f'Serving preprocessing differs from training: {name}')
        if (checkpoint['model_config']['edge_dim'] != len(self.manifest['feature_names'])
                or checkpoint['model_config']['in_dim'] != len(self.manifest['node_feature_names'])):
            raise ValueError('Checkpoint dimensions differ from serving schema')
        self.device = device

    def inputs(self, request, history):
        """Read the complete committed prefix, not just one account or a short window."""
        request = normalize_event(request)
        if request is None or request['event_type'] not in REQUESTS:
            raise ValueError('Inference target must be a request event')
        at = microseconds(request['occurred_at'])
        events, seen = [], {}
        previous = None
        for raw in history:
            raw = unwrap_record(raw)
            if raw is None:
                continue
            # BE may already have persisted the target and its later settlement.
            raw_at = raw.get('occurred_at', raw.get('occurredAt', raw.get('created_at')))
            if microseconds(raw_at) >= at:
                continue
            event = normalize_event(raw)
            if event is None:
                continue
            timestamp = microseconds(event['occurred_at'])
            if previous is not None and timestamp < previous:
                raise ValueError('History must be chronological')
            previous = timestamp
            identity = event['event_id']
            if identity in seen:
                if event != seen[identity]:
                    raise ValueError('Conflicting duplicate history event')
                continue
            if event['aggregate_id'] == request['aggregate_id']:
                raise ValueError('Target transaction already has prior history; replay its original request time')
            seen[identity] = event
            events.append(event)
        # Reproduce the conservative graph snapshot policy used by TemporalBatches.
        width = self.bundle['snapshot_seconds'] * 1000000
        bucket = at // width
        cutoff = next((microseconds(e['occurred_at']) for e in events
                       if e['event_type'] in REQUESTS and microseconds(e['occurred_at']) // width == bucket), at)
        nodes, mapping = request_account_catalog(self.nodes, request, events, cutoff)
        state = State(self.options, self.zone)
        for _ in replay(events, state):
            pass
        features = np.asarray([state.score(request)], dtype='float32')
        prep = self.preprocessing
        columns = prep['numeric_indices']
        features[:, columns] = (features[:, columns] - np.asarray(prep['mean'])) / np.asarray(prep['scale'])

        def node_id(identity, timestamp):
            i = mapping.get(identity, 0)
            return i if nodes[i]['created_at_us'] <= timestamp else 0

        graph = []
        for event in events:
            timestamp = microseconds(event['occurred_at'])
            if timestamp >= cutoff or event['event_type'] not in REQUESTS | COMPLETED:
                continue
            p = event['payload']
            src, dst = node_id(p['sourceAccountId'], timestamp), node_id(p['destinationAccountId'], timestamp)
            if p.get('transactionType') == 'CASH_IN':
                src, dst = dst, src
            graph.append((timestamp, src, dst, int(event['event_type'] in COMPLETED), np.log1p(p['amount'])))
        static = np.zeros((len(nodes), len(NODE_TYPES)), dtype='float32')
        for i, node in enumerate(nodes):
            static[i, NODE_TYPES.index(node['type'])] = 1
        arrays = {'node_features': static,
                  'node_created_at_us': np.asarray([n['created_at_us'] for n in nodes]),
                  'graph_timestamps_us': np.asarray([g[0] for g in graph], dtype='int64'),
                  'graph_edge_index': np.asarray([[g[1] for g in graph], [g[2] for g in graph]], dtype='int64'),
                  'graph_kind': np.asarray([g[3] for g in graph], dtype='uint8'),
                  'graph_log_amount': np.asarray([g[4] for g in graph], dtype='float32')}
        context = graph_context(arrays, self.options, self.node_preprocessing, cutoff, len(graph))
        p = request['payload']
        target = np.asarray([[node_id(p['sourceAccountId'], at)], [node_id(p['destinationAccountId'], at)]], dtype='int64')
        target[~context['node_available_mask'][target]] = 0
        if not np.isfinite(features).all() or not np.isfinite(context['x']).all():
            raise ValueError('Nonfinite inference input')
        return {**context, 'target_edge_index': target, 'edge_features': features}

    def predict(self, request, history):
        import torch
        event = normalize_event(request)
        context = self.inputs(request, history)
        tensors = {k: torch.from_numpy(context[k]).to(device=self.device,
                    dtype=torch.float32 if k in ('x', 'edge_features') else torch.long)
                   for k in ('x', 'edge_index', 'target_edge_index', 'edge_features')}
        with torch.inference_mode():
            score = float(torch.sigmoid(self.model(**tensors)).item())
        if not np.isfinite(score):
            raise ValueError('Nonfinite risk score')
        return build_response(transaction_id=request.get('transaction_id', event['aggregate_id']),
                              risk_score=score, threshold=self.bundle['threshold'],
                              model_version=self.bundle['model_version'],
                              feature_version=self.bundle['feature_version'])
