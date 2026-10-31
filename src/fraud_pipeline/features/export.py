"""Export causal tensors plus a compact temporal graph log, without Torch."""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..ingestion.request_loader import file_hash, jsonl_rows, verify_validation
from .causal import State, TYPES, COMPLETED, microseconds, names, replay
from .nodes import NODE_DYNAMIC_NAMES, dynamic_features

NODE_TYPES = ('unknown', 'account', 'merchant', 'cash')
SPLITS = ('train', 'validation', 'test')


def export_features(source, validation_report, splits, output, options, *, overwrite=False):
    import numpy as np
    source, validation_report, splits, output = map(lambda p: Path(p).resolve(),
                                                   (source, validation_report, splits, output))
    manifest, hashes, _ = verify_validation(source, validation_report)
    split_manifest = json.loads((splits / 'manifest.json').read_text())
    if split_manifest['source']['directory'] != str(source) or split_manifest['source']['sha256'] != hashes:
        raise ValueError('Split source provenance disagrees with audit')
    if split_manifest['validation']['report_sha256'] != file_hash(validation_report):
        raise ValueError('Split validation checksum disagrees with current report; re-export split')
    if (any(output == p or output in p.parents or p in output.parents for p in (source, splits))
            or output == validation_report or output in validation_report.parents):
        raise ValueError('Feature output must be separate from input directories')
    if output.exists() and not overwrite:
        raise ValueError('Feature output already exists; use --overwrite or another output')
    state = State(options, datetime.fromisoformat(manifest['start_at']).tzinfo)
    input_hashes = {str(splits / 'manifest.json'): file_hash(splits / 'manifest.json'),
                    str(validation_report): file_hash(validation_report)}
    for name in SPLITS:
        path = splits / split_manifest['splits'][name]['file']
        if file_hash(path) != split_manifest['splits'][name]['sha256']:
            raise ValueError(f'Split checksum mismatch: {name}')
        input_hashes[str(path)] = file_hash(path)
    count = split_manifest['total_samples']
    schema = names(options)
    nodes = [{'id': '__unknown__', 'type': 'unknown', 'created_at_us': -1}]
    for table, kind in [('accounts', 'account'), ('counterparties', None)]:
        for _, row in jsonl_rows(source / f'{table}.jsonl'):
            nodes.append({'id': row['_id'], 'type': kind or row['type'],
                          'created_at_us': microseconds(row['created_at'])})
    nodes[1:] = sorted(nodes[1:], key=lambda n: (n['created_at_us'], n['id']))
    mapping = {node['id']: i for i, node in enumerate(nodes)}
    if len(mapping) != len(nodes):
        raise ValueError('Duplicate node identity or reserved __unknown__ ID')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.features-build-', dir=output.parent) as temp:
        staged = Path(temp) / 'features'
        staged.mkdir()
        def allocate(name, dtype, shape):
            return np.lib.format.open_memmap(staged / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
        features = allocate('transaction_features', 'float32', (count, len(schema)))
        target_edges = allocate('target_edge_index', 'int64', (2, count))
        flow_edges = allocate('flow_edge_index', 'int64', (2, count))
        targets = allocate('labels', 'uint8', (count,))
        codes = allocate('split_codes', 'uint8', (count,))
        times = allocate('request_timestamps_us', 'int64', (count,))
        prefixes = allocate('graph_prefix', 'int64', (count,))
        graph = {name: allocate('.' + name, dtype, shape) for name, dtype, shape in [
            ('graph_edge_index', 'int64', (2, 2 * count)), ('graph_timestamps_us', 'int64', (2 * count,)),
            ('graph_kind', 'uint8', (2 * count,)), ('graph_type', 'uint8', (2 * count,)),
            ('graph_log_amount', 'float32', (2 * count,))]}
        graph_count = 0
        def node_id(identity, at):
            index = mapping.get(identity, 0)
            return index if nodes[index]['created_at_us'] <= at else 0
        def record_graph(event):
            nonlocal graph_count
            p, at = event['payload'], microseconds(event['occurred_at'])
            kind = 'TRANSFER' if event['event_type'].startswith('Transfer') else p.get('transactionType', 'unknown')
            src, dst = node_id(p['sourceAccountId'], at), node_id(p['destinationAccountId'], at)
            if kind == 'CASH_IN':
                src, dst = dst, src
            graph['graph_edge_index'][:, graph_count] = [src, dst]
            graph['graph_timestamps_us'][graph_count] = at
            graph['graph_kind'][graph_count] = int(event['event_type'] in COMPLETED)
            graph['graph_type'][graph_count] = TYPES.index(kind if kind in TYPES else 'unknown')
            graph['graph_log_amount'][graph_count] = np.log1p(p['amount'])
            graph_count += 1
        def split_rows():
            for code, name in enumerate(SPLITS):
                for _, row in jsonl_rows(splits / split_manifest['splits'][name]['file']):
                    yield code, row
        samples = iter(split_rows())
        def events():
            for line, event in jsonl_rows(source / 'events.jsonl'):
                event['_source_line'] = line
                yield event
        actual = 0
        with (staged / 'requests.jsonl').open('w') as trace:
            for event, values, prefix in replay(events(), state, record_graph):
                sample = next(samples, None)
                if sample is None:
                    raise ValueError('More requests than split samples')
                code, row = sample
                p, at = event['payload'], microseconds(event['occurred_at'])
                kind = 'TRANSFER' if event['event_type'] == 'TransferCreated' else p['transactionType']
                if (row['event_id'] != event['event_id'] or row['transaction_id'] != event['aggregate_id']
                    or microseconds(row['timestamp']) != at or row['source_event_line'] != event['_source_line']
                    or any(row[k] != p[v] for k, v in [('src_acc', 'sourceAccountId'), ('dst_acc', 'destinationAccountId'),
                                                     ('amount', 'amount'), ('fee', 'fee'), ('channel', 'channel')])
                    or row['type'] != kind):
                    raise ValueError('Request/split alignment mismatch')
                for key in ('authentication_method','authentication_verified','device_id','platform'):
                    if row.get(key)!=p.get(key):raise ValueError('Optional request context/split mismatch')
                features[actual] = values
                src, dst = node_id(row['src_acc'], at), node_id(row['dst_acc'], at)
                target_edges[:, actual] = [src, dst]
                flow_edges[:, actual] = [dst, src] if kind == 'CASH_IN' else [src, dst]
                targets[actual], codes[actual], times[actual], prefixes[actual] = row['label'], code, at, prefix
                trace.write(json.dumps({'row': actual, 'event_id': row['event_id'], 'transaction_id': row['transaction_id'],
                                        'source_event_line': row['source_event_line'], 'timestamp': row['timestamp']}) + '\n')
                actual += 1
        if actual != count or next(samples, None) is not None:
            raise ValueError('Request/split count mismatch')
        ntrain = split_manifest['splits']['train']['count']
        if not np.all(codes[:ntrain] == 0) or np.any(codes[ntrain:] == 0):
            raise ValueError('Train masks are not chronological')
        # Learn only numeric transforms; categorical vocabularies are fixed public contracts.
        numeric = [i for i, name in enumerate(schema) if name.startswith('log_')]
        mean, variance, seen = np.zeros(len(numeric)), np.zeros(len(numeric)), 0
        chunk = options['chunk_size']
        for start in range(0, ntrain, chunk):
            part = np.asarray(features[start:min(start + chunk, ntrain), numeric], dtype='float64')
            batch_n, batch_mean = len(part), part.mean(axis=0)
            delta = batch_mean - mean
            total = seen + batch_n
            variance += ((part - batch_mean) ** 2).sum(axis=0) + delta ** 2 * seen * batch_n / total
            mean += delta * batch_n / total
            seen = total
        scale = np.sqrt(variance / seen)
        scale[scale < 1e-12] = 1.
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            part = features[start:stop].copy()
            part[:, numeric] = (part[:, numeric] - mean) / scale
            if not np.isfinite(part).all():
                raise ValueError('Nonfinite transaction features')
            features[start:stop] = part
        for value in (features, target_edges, flow_edges, targets, codes, times, prefixes):
            value.flush()
        node_samples = np.unique(np.linspace(0, ntrain - 1, min(ntrain, options['node_scaler_samples']), dtype='int64'))
        node_sum, node_square, node_fit_rows = np.zeros(6), np.zeros(6), 0
        for index in node_samples:
            at = int(times[index])
            dynamic = dynamic_features(graph, at, int(prefixes[index]), len(nodes), options['node_history_window_seconds'])
            available = np.asarray([node['created_at_us'] <= at for node in nodes])
            available[0] = False
            values = np.asarray(dynamic[available, :6], dtype='float64')
            node_sum += values.sum(axis=0)
            node_square += (values ** 2).sum(axis=0)
            node_fit_rows += len(values)
        if not node_fit_rows:
            raise ValueError('No available training nodes for preprocessing')
        node_mean = node_sum / node_fit_rows
        node_scale = np.sqrt(np.maximum(0, node_square / node_fit_rows - node_mean ** 2))
        node_scale[node_scale < 1e-12] = 1
        node_preprocessing = {'mean': node_mean.tolist(), 'scale': node_scale.tolist(),
                              'numeric_indices': list(range(6)), 'fit_split': 'train',
                              'sampled_request_indices': node_samples.tolist(), 'fit_rows': node_fit_rows,
                              'method': 'Uniformly sampled train request contexts; available known nodes only.'}
        (staged / 'node_preprocessing.json').write_text(json.dumps(node_preprocessing, indent=2) + '\n')
        for name, value in graph.items():
            value.flush()
            np.save(staged / (name + '.npy'), value[:, :graph_count] if value.ndim == 2 else value[:graph_count])
            # Windows cannot delete an open NumPy memory-mapped file.
            value._mmap.close()
            (staged / ('.' + name + '.npy')).unlink()
        node_features = np.zeros((len(nodes), len(NODE_TYPES)), dtype='float32')
        for i, node in enumerate(nodes):
            node_features[i, NODE_TYPES.index(node['type'])] = 1
        np.save(staged / 'node_features.npy', node_features)
        np.save(staged / 'node_created_at_us.npy', np.asarray([n['created_at_us'] for n in nodes], dtype='int64'))
        for code, name in enumerate(SPLITS):
            np.save(staged / (name + '_mask.npy'), codes == code)
        # Release mapped outputs before moving the staging directory on Windows.
        for value in (features, target_edges, flow_edges, targets, codes, times, prefixes):
            value.flush()
            value._mmap.close()
        (staged / 'nodes.json').write_text(json.dumps(nodes, indent=2) + '\n')
        preprocessing = {'numeric_indices': numeric, 'mean': mean.tolist(), 'scale': scale.tolist(),
                         'fit_split': 'train', 'fit_count': seen, 'type_vocabulary': TYPES, 'channel_vocabulary': ('mobile', 'web', 'desktop', 'unknown'),
                         'missing_policy': 'No history uses zero plus has_previous/has_completed_history/cold_start flags. Required fields are validated, not imputed.'}
        (staged / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2) + '\n')
        verify_validation(source, validation_report)
        for path, digest in input_hashes.items():
            if file_hash(Path(path)) != digest:
                raise ValueError('Input changed during feature export')
        result = {'schema_version': 3 if options.get('mean_window_seconds') or options.get('include_request_context') else 2, 'source': str(source), 'source_sha256': hashes, 'input_sha256': input_hashes,
                  'request_count': count, 'node_count': len(nodes), 'graph_event_count': graph_count,
                  'feature_names': schema, 'node_static_feature_names': NODE_TYPES,
                  'node_feature_names': [*NODE_TYPES, *NODE_DYNAMIC_NAMES], 'options': options,
                  'request_context_policy': 'Optional request-time snake_case fields authentication_method/authentication_verified/device_id/platform. Missing is encoded with flags and unknown category; device_id is used only for prior novelty, never raw numeric input. Current posted_at excluded.',
                  'mean_policy': 'Completed amount mean per source+type, configurable window; separate support/reliability flags. Median reference remains separate.',
                  'timezone': datetime.fromisoformat(manifest['start_at']).strftime('%z'),
                  'code_sha256': {p.name: file_hash(p) for p in [Path(__file__), *[Path(__file__).with_name(n) for n in ('causal.py','dataset.py','history.py','nodes.py','request_context.py')]]},
                  'split_counts': {name: split_manifest['splits'][name]['count'] for name in SPLITS},
                  'graph_kind': {'attempt': 0, 'settled': 1}, 'graph_type': TYPES,
                  'time_policy': 'Requests at t see only timestamps < t. All same-timestamp requests score before group observations.',
                  'graph_policy': 'Money-flow direction; CASH_IN cash→account. No explicit self loops; GAT layers add them. Per-request [t-window,t) graph, deduplicated pairs, optional reverse edges.',
                  'target_policy': 'target_edge_index preserves raw source/destination roles; flow_edge_index follows money flow.',
                  'node_policy': 'Type one-hot + exact settled incoming/outgoing count, unique degree, median amount and history flags at each cutoff. Node numeric scaler fits sampled train contexts only. Unknown buckets; created_at gates availability.',
                  'batch_policy': 'Exact request-specific graph context; different cutoffs must not share a later snapshot.',
                  'excluded': ['labels_from_state', 'scenario', 'generation_profiles', 'credit', 'loan', 'identity', 'balances', 'current_terminal_result'],
                  'limitations': 'Node contexts reconstructed exactly; training throughput requires measurement. Synthetic funding/activity limitations persist. No training in Phase 3.'}
        result['files'] = {p.name: file_hash(p) for p in sorted(staged.iterdir()) if p.is_file()}
        (staged / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        backup = None
        if output.exists():
            if not (output / 'manifest.json').is_file():
                raise ValueError('Refusing overwrite of unrecognized output')
            backup = output.with_name(output.name + '.previous-' + uuid4().hex[:8])
            output.rename(backup)
        try:
            os.replace(staged, output)
        except OSError:
            if backup is not None:
                backup.rename(output)
            raise
    return result
