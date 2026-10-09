"""Compare causal amount/behavior/graph diagnostic rules on exported splits."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.artifacts import defaults
from fraud_pipeline.evaluation.dataset_baseline import CausalRules, binary_metrics, ranking_metrics
from fraud_pipeline.ingestion.request_loader import file_hash, jsonl_rows, verify_validation
from fraud_pipeline.validation.core import timestamp


def assess(splits, validation_report, output):
    manifest = json.loads((splits/'manifest.json').read_text())
    source = Path(manifest['source']['directory'])
    _, hashes, audit = verify_validation(source, validation_report)
    if hashes != manifest['source']['sha256']:
        raise ValueError('Split source differs from current dataset audit')
    labels = {r['transaction_id']: (int(r['is_anomaly']), r['scenario_id']) for _, r in jsonl_rows(source/'labels.jsonl')}
    state = CausalRules()
    data = {name: defaultdict(list) for name in ['validation', 'test']}
    count = Counter()
    for name in ['train', 'validation', 'test']:
        stats = manifest['splits'][name]
        path = splits/stats['file']
        if file_hash(path) != stats['sha256']:
            raise ValueError('Split output checksum mismatch')
        for _, row in jsonl_rows(path):
            label, scenario = labels[row['transaction_id']]
            if row['label'] != label:
                raise ValueError('Split and sidecar targets differ')
            scores = state.score_then_observe(row, timestamp(row['timestamp']).timestamp())
            count[name] += 1
            if name != 'train':
                for rule, score in scores.items():
                    data[name][rule].append((score, label, scenario))
        if count[name] != stats['count']:
            raise ValueError('Split row count mismatch')
    results = {}
    for rule in data['validation']:
        validation = data['validation'][rule]
        test = data['test'][rule]
        # A fixed candidate grid; selection uses validation labels only.
        candidates = [i/10 for i in range(1, 121)]
        best = max((binary_metrics(validation, t) for t in candidates), key=lambda r: r['f1'])
        scenario_totals, caught = Counter(), Counter()
        for score, label, scenario in test:
            if label:
                scenario_totals[scenario] += 1
                caught[scenario] += score >= best['threshold']
        results[rule] = {'validation': dict(best, **ranking_metrics(validation)),
            'test': dict(binary_metrics(test, best['threshold']), **ranking_metrics(test)),
            'test_scenario_recall': {s: {'total': n, 'detected': caught[s], 'recall': caught[s]/n} for s, n in scenario_totals.items()}}
    result = {'seed': manifest['source']['seed'], 'scenario_profile': audit['manifest'].get('scenario_profile', 'amount-only-v2'),
        'dataset_sha256': hashes, 'split_counts': dict(count), 'rules': results,
        'overall': audit['summary']['overall'], 'monthly': audit['summary']['monthly'], 'graph': audit['summary']['graph'],
        'scenario_counts': dict(Counter(s for _, s in labels.values())),
        'method': 'Score before observing current request; carry unlabeled history across time splits; threshold selected on validation only.',
        'limitations': 'Rules follow synthetic scenario assumptions. Request graph includes attempts, not only completed money flow. This is not GATv2 performance or real-bank validation. Generation profiles and scenario names are never feature inputs.'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    for rule, r in results.items():
        t = r['test']
        print(f"{rule}: precision={t['precision']:.4f}, recall={t['recall']:.4f}, F1={t['f1']:.4f}, AUC={t['roc_auc']:.4f}, AP={t['average_precision']:.4f}", flush=True)
    return result


if __name__ == '__main__':
    paths = defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits', type=Path, default=paths['splits'])
    ap.add_argument('--validation-report', type=Path, default=paths['validation'])
    ap.add_argument('--output', type=Path, default=paths['baseline'])
    args = ap.parse_args()
    assess(args.splits, args.validation_report, args.output)
