"""Phase 1 dataset audit (stdlib only)."""
from datetime import datetime, time
from pathlib import Path

from .checks import validate_tables
from .core import Audit, integer, read_dataset, timestamp
from .report import FEATURE_POLICY, summaries


def validate_dataset(root: Path):
    root = Path(root)
    audit = Audit()
    tables, manifest, hashes = read_dataset(root, audit)
    period = None
    try:
        start = timestamp(manifest['start_at'])
        end = datetime.combine(datetime.fromisoformat(manifest['end_date_exclusive']).date(), time(), start.tzinfo)
        if end <= start:
            raise ValueError('Empty simulation period')
        if not integer(manifest.get('days'), 1) or (end.date() - start.date()).days != manifest['days']:
            raise ValueError('Manifest days and dates disagree')
        period = (start, end)
    except (KeyError, ValueError, TypeError) as exc:
        audit.check(False, 'manifest_period', 'manifest', str(exc))
    transactions, labels, events, ledger = validate_tables(tables, manifest, audit)
    if period:
        for e in events:
            audit.check(period[0] <= timestamp(e['occurred_at']) < period[1], 'event_period', e['event_id'], 'Event outside simulation period')
        for tx in transactions.values():
            audit.check(period[0] <= timestamp(tx['created_at']) < period[1], 'transaction_period', tx['transaction_id'], 'Request outside simulation period')
    for kind in {e['event_type'] for e in events}:
        if kind == 'LoanPaymentPosted':
            audit.warning('loan_final_state', 'loans', 'Loan snapshots/outstanding principal are final-state assumptions, not replay-derived as-of balances.')
    summary = summaries(transactions, labels, events, manifest)
    if not transactions:
        audit.check(False, 'empty_dataset', 'transactions', 'No valid transactions')
    for kind, data in summary['by_type'].items():
        if data['count'] and data['anomaly'] < 5:
            audit.warning('sparse_type_anomalies', kind, f"Only {data['anomaly']} anomalies; per-type metrics will be unstable.")
    for preview in summary['time_split_preview']:
        if preview['normal'] == 0 or preview['anomaly'] == 0:
            audit.warning('split_class_missing', preview['name'], 'Time split preview lacks one class; adjust generator size before evaluation.')
    errors = sum(i['severity'] == 'error' for i in audit.issues)
    warnings = sum(i['severity'] == 'warning' for i in audit.issues)
    return {'status': 'failed' if errors else 'passed_with_warnings' if warnings else 'passed',
            'error_count': errors, 'warning_count': warnings, 'issues': audit.issues,
            'dataset': str(root.resolve()), 'manifest': manifest, 'sha256': hashes,
            'feature_policy': FEATURE_POLICY, 'ledger': ledger, 'summary': summary}
