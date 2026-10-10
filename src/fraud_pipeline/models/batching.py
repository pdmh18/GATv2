"""Conservative temporal graph snapshots shared only within a causal time bucket."""
from pathlib import Path
import json
from ..ingestion.request_loader import file_hash


class TemporalBatches:
    def __init__(self, dataset, snapshot_seconds, batch_size, cache, *, allow_test=False):
        if type(snapshot_seconds) is not int or snapshot_seconds <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Snapshot seconds and batch size must be positive integers')
        self.dataset, self.seconds, self.batch_size = dataset, snapshot_seconds, batch_size
        self.allow_test = allow_test
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        identity={'feature_manifest_sha256':file_hash(dataset.directory/'manifest.json') if hasattr(dataset,'directory') else 'in-memory-test',
                  'snapshot_seconds':snapshot_seconds,'schema_version':1}
        metadata=self.cache/'manifest.json'
        if metadata.exists() and json.loads(metadata.read_text())!=identity:
            raise ValueError('Snapshot cache belongs to another feature artifact or policy')
        metadata.write_text(json.dumps(identity,indent=2)+'\n')

    def groups(self, split):
        import numpy as np
        if split not in ('train', 'validation') and not (split == 'test' and self.allow_test):
            raise ValueError('Phase 4 batches permit train/validation only')
        indices = np.flatnonzero(self.dataset.arrays[split + '_mask'])
        buckets = self.dataset.arrays['request_timestamps_us'][indices] // (self.seconds * 1000000)
        boundaries = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(indices)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group = indices[start:end]
            if len(group):
                yield group

    def context(self, first):
        import numpy as np
        path = self.cache / f'{int(first):09d}.npz'
        if path.exists():
            if not path.with_suffix('.sha256').exists() or path.with_suffix('.sha256').read_text()!=file_hash(path):
                raise ValueError('Snapshot cache checksum mismatch')
            with np.load(path, allow_pickle=False) as saved:
                context = {name: saved[name] for name in saved.files}
        else:
            full = self.dataset.context(int(first))
            context = {name: full[name] for name in ('x','edge_index','node_available_mask')}
            context['timestamp_us'] = np.asarray(full['timestamp_us'], dtype='int64')
            np.savez(path, **context)
            path.with_suffix('.sha256').write_text(file_hash(path))
        if int(context['timestamp_us']) != int(self.dataset.arrays['request_timestamps_us'][first]):
            raise ValueError('Cached context timestamp mismatch')
        return context

    def iterate(self, split):
        import numpy as np
        arrays = self.dataset.arrays
        for group in self.groups(split):
            context = self.context(group[0])
            cutoff = int(context['timestamp_us'])
            for start in range(0,len(group),self.batch_size):
                indices = group[start:start+self.batch_size]
                if np.any(arrays['request_timestamps_us'][indices] < cutoff):
                    raise ValueError('Snapshot is later than target request')
                targets = arrays['target_edge_index'][:,indices].copy()
                targets[~context['node_available_mask'][targets]] = 0
                yield {'x':context['x'], 'edge_index':context['edge_index'],
                       'target_edge_index':targets, 'edge_features':arrays['transaction_features'][indices].copy(),
                       'labels':arrays['labels'][indices].copy(), 'indices':indices,
                       'cutoff_us':cutoff}
