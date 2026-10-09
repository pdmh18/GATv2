"""Memory-mapped NumPy context loader; Torch conversion belongs to training."""
import json
from pathlib import Path
from ..ingestion.request_loader import file_hash
from .nodes import dynamic_features


class TemporalDataset:
    def __init__(self, directory, *, verify=True):
        import numpy as np
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if verify:
            for name, digest in self.manifest['files'].items():
                if Path(name).name != name or file_hash(self.directory / name) != digest:
                    raise ValueError(f'Feature artifact checksum mismatch: {name}')
        self.arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
                       for p in self.directory.glob('*.npy')}
        n = self.manifest['request_count']
        if self.arrays['transaction_features'].shape != (n, len(self.manifest['feature_names'])):
            raise ValueError('Transaction feature shape mismatch')
        if self.arrays['target_edge_index'].shape != (2, n):
            raise ValueError('Target edge shape mismatch')
        if self.arrays['graph_edge_index'].shape != (2, self.manifest['graph_event_count']):
            raise ValueError('Graph timeline shape mismatch')
        self.node_preprocessing = (json.loads((self.directory/'node_preprocessing.json').read_text())
                                   if self.manifest['schema_version'] >= 2 else None)

    def __len__(self):
        return self.manifest['request_count']

    def context(self, index):
        """Return causal model inputs only; labels and split masks stay separate."""
        import numpy as np
        if not 0 <= index < len(self):
            raise IndexError(index)
        a, options = self.arrays, self.manifest['options']
        at, end = int(a['request_timestamps_us'][index]), int(a['graph_prefix'][index])
        times = a['graph_timestamps_us']
        if not 0 <= end <= len(times) or (end and times[end - 1] >= at) or (end < len(times) and times[end] < at):
            raise ValueError('Graph prefix violates strict causal cutoff')
        start = int(np.searchsorted(times[:end], at - options['graph_window_seconds'] * 1000000, side='left'))
        edges = a['graph_edge_index'][:, start:end]
        kinds = a['graph_kind'][start:end]
        selection = np.ones(len(kinds), dtype=bool)
        if options['graph_history'] != 'both':
            selection &= kinds == int(options['graph_history'] == 'settled')
        edges = np.asarray(edges[:, selection])
        edges = edges[:, edges[0] != edges[1]]
        if options['bidirectional_graph']:
            edges = np.concatenate([edges, edges[::-1]], axis=1)
        edges = np.unique(edges, axis=1)
        node_features = a['node_features'].copy()
        available = a['node_created_at_us'] <= at
        if self.node_preprocessing is not None:
            dynamic = dynamic_features(a, at, end, self.manifest['node_count'], options['node_history_window_seconds'])
            prep = self.node_preprocessing
            dynamic[:, :6] = (dynamic[:, :6] - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
            node_features = np.concatenate([node_features, dynamic], axis=1)
        node_features[~available] = 0
        return {'x': node_features, 'edge_index': edges,
                'target_edge_index': a['target_edge_index'][:, index:index + 1].copy(),
                'flow_edge_index': a['flow_edge_index'][:, index:index + 1].copy(),
                'edge_features': a['transaction_features'][index:index + 1].copy(),
                'node_available_mask': np.asarray(available), 'timestamp_us': at,
                'graph_interval': [start, end]}
