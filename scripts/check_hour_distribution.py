"""Audit hour overlap using labels/profiles only for offline dataset analysis."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fraud_pipeline.artifacts import defaults, seed_paths
from fraud_pipeline.ingestion.request_loader import jsonl_rows, verify_validation


def inspect(source, validation):
    manifest, hashes, _ = verify_validation(source, validation)
    labels = {r['transaction_id']: r for _, r in jsonl_rows(source / 'labels.jsonl')}
    profiles = {r['account_id']: r for _, r in jsonl_rows(source / 'behavior_profiles.jsonl')}
    hours = {'normal': [0] * 24, 'anomaly': [0] * 24}
    scenarios = defaultdict(Counter)
    for _, tx in jsonl_rows(source / 'transactions.jsonl'):
        label = labels[tx['transaction_id']]
        hour = datetime.fromisoformat(tx['created_at']).hour
        hours['anomaly' if label['is_anomaly'] else 'normal'][hour] += 1
        preferred = profiles[tx['source']['account_id']]['preferred_hours']
        scenarios[label['scenario_id']]['preferred' if hour in preferred else 'outside'] += 1
    return {'seed': manifest['seed'], 'scenario_profile': manifest['scenario_profile'],
            'dataset_sha256': hashes, 'hour_counts': hours,
            'empty_hours': {k: [h for h, n in enumerate(v) if n == 0] for k, v in hours.items()},
            'scenario_source_hours': dict(scenarios),
            'notes': ['Graph schedules use hub hours; donor source preferences can differ.',
                      'Overlap removes a zero-support shortcut but does not establish real-bank calibration.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=defaults()['run_root'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261002, 20261003, 20261004])
    args = parser.parse_args()
    results = [inspect(seed_paths(args.output_root, seed)['raw'],
                       seed_paths(args.output_root, seed)['validation'] / 'validation.json') for seed in args.seeds]
    output = args.output_root / 'hour-distribution.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        print(result['seed'], 'empty hours:', result['empty_hours'],
              'anomaly 14–16:', result['hour_counts']['anomaly'][14:17])


if __name__ == '__main__':
    main()
