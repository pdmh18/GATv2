"""Verify tensor alignment, masks, train scaling and causal graph cutoffs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fraud_pipeline.config import load_config, project_path
from fraud_pipeline.features.dataset import TemporalDataset


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=project_path(load_config('configs/dataset.json')['dataset']['features_output']))
    args = parser.parse_args()
    data = TemporalDataset(args.features)
    a, manifest = data.arrays, data.manifest
    n, d = len(data), len(manifest['feature_names'])
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(a['labels'].shape == (n,) and np.isin(a['labels'], [0, 1]).all(), 'Invalid labels')
    require(a['flow_edge_index'].shape == (2, n), 'Invalid money-flow target shape')
    require((np.diff(a['request_timestamps_us']) >= 0).all(), 'Request timestamps are not monotonic')
    require((np.diff(a['graph_timestamps_us']) >= 0).all(), 'Graph timestamps are not monotonic')
    require((a['target_edge_index'] >= 0).all() and (a['target_edge_index'] < manifest['node_count']).all(), 'Target node bounds')
    require((a['graph_edge_index'] >= 0).all() and (a['graph_edge_index'] < manifest['node_count']).all(), 'Graph node bounds')
    prefix, times = a['graph_prefix'], a['request_timestamps_us']
    expected = np.searchsorted(a['graph_timestamps_us'], times, side='left')
    require(np.array_equal(prefix, expected), 'Future/current timestamp enters graph prefix')
    masks = [a[name + '_mask'] for name in ('train', 'validation', 'test')]
    require(all(mask.shape == (n,) and mask.dtype == bool for mask in masks), 'Invalid mask shape/type')
    require(np.all(sum(mask.astype(int) for mask in masks) == 1), 'Masks overlap or omit requests')
    for code, name in enumerate(('train', 'validation', 'test')):
        require(np.array_equal(masks[code], a['split_codes'] == code), 'Mask/split code mismatch')
        require(int(masks[code].sum()) == manifest['split_counts'][name], 'Split count mismatch')
    for start in range(0, n, 10000):
        require(np.isfinite(a['transaction_features'][start:start + 10000]).all(), 'Nonfinite features')
    prep = json.loads((args.features / 'preprocessing.json').read_text())
    train = a['transaction_features'][:prep['fit_count'], prep['numeric_indices']]
    require(np.max(np.abs(train.mean(axis=0, dtype='float64'))) < 1e-5, 'Training numeric means differ from zero')
    sampled = np.unique(np.concatenate([np.linspace(0, n - 1, min(n, 1000), dtype=int),
                                        [int(mask.sum()) for mask in masks[:1]]]))
    graph_sizes = []
    account_unique = []
    isolated_sources = 0
    for i in sampled:
        context = data.context(int(i))
        require(context['x'].shape == (manifest['node_count'], len(manifest['node_feature_names'])), 'Node feature shape mismatch')
        require(np.isfinite(context['x']).all(), 'Nonfinite node features')
        require(context['edge_features'].shape == (1, d), 'Context transaction feature shape mismatch')
        graph_sizes.append(context['edge_index'].shape[1])
        src = int(context['target_edge_index'][0,0])
        isolated_sources += not np.any(context['edge_index'] == src)
        account_mask = a['node_features'][:,1] == 1
        account_unique.append(len(np.unique(context['x'][account_mask],axis=0)))
    result = {'status': 'passed', 'request_count': n, 'node_count': manifest['node_count'],
              'transaction_feature_count': d, 'node_feature_count':len(manifest['node_feature_names']),
              'graph_event_count': manifest['graph_event_count'],
              'sampled_contexts': len(sampled), 'sampled_graph_edges_min_max': [min(graph_sizes), max(graph_sizes)],
              'sampled_source_isolated_fraction':isolated_sources/len(sampled),
              'sampled_unique_account_vectors_min_max':[min(account_unique),max(account_unique)],
              'split_counts': manifest['split_counts'], 'train_fit_count': prep['fit_count'],
              'checks': ['Artifact checksums', 'Tensor shapes and finite values', 'Node bounds',
                         'Masks and counts', 'All-request strict graph cutoffs', 'Train-only centering', 'Sampled context loading']}
    # This audit is deliberately outside features/ so its creation does not invalidate the artifact manifest.
    path = args.features.parent / 'validation' / 'features.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
