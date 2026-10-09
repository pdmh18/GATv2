"""Corruption tests use an independent hand-built ledger, not the generator."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import NAMESPACE_OID, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud_pipeline.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    start = '2025-01-01T00:00:00+07:00'
    tables = {name: [] for name in ['accounts', 'customers', 'counterparties', 'transactions', 'events', 'labels',
                                   'transfer_events', 'loan_packages', 'loan_applications', 'loans']}
    for aid in ['A', 'B']:
        credit = {'score': 700, 'bad_debt': False, 'updated_at': start,
                  'history': [{'as_of': '2025-01-01', 'score': 700, 'payment_status': 'on_time'}]}
        customer = {'_id': 'C' + aid, 'customer_code': aid, 'profile': {}, 'identity': {}, 'credit_profile': credit,
                    'kyc_status': 'verified', 'status': 'active', 'created_at': start, 'updated_at': start}
        tables['customers'].append(customer)
        tables['accounts'].append({'_id': aid, 'account_no': aid, 'customer_id': 'C' + aid,
                                   'balance': {'current': 500}, 'created_at': start, 'currency': 'VND'})
        for kind, aggregate, identity, sequence, payload in [
            ('AccountOpened', 'account', aid, 1, {'account_id': aid, 'customer_id': 'C'+aid, 'account_no': aid, 'opening_balance': 500}),
            ('CustomerProfileCreated', 'customer', 'C'+aid, 1, customer),
            ('CreditProfileRecorded', 'customer', 'C'+aid, 2, credit)]:
            tables['events'].append({'event_id': f'{kind}-{aid}', 'aggregate_type': aggregate, 'aggregate_id': identity,
                                    'event_type': kind, 'schema_version': 2, 'occurred_at': start,
                                    'sequence': sequence, 'payload': payload, 'labels': None})
    tables['counterparties'] = [{'_id': 'M', 'type': 'merchant', 'created_at': start}, {'_id': 'CASH', 'type': 'cash', 'created_at': start}]
    cases = [
        ('TRANSFER', 'A', 'B', 100, 5, 500, 395, 500, 600, 2, False),
        ('CASH_IN', 'A', 'CASH', 200, 0, 395, 595, None, None, 4, True),
        ('CASH_OUT', 'B', 'CASH', 70, 3, 600, 527, None, None, 8, False),
        ('PAYMENT', 'A', 'M', 40, 2, 595, 553, None, None, 12, False),
        ('DEBIT', 'B', 'M', 30, 1, 527, 496, None, None, 17, True),
        ('TRANSFER', 'A', 'B', 1000, 5, 553, 553, 496, 496, 18, True)]
    for n, (kind, src, dst, amount, fee, before, after, db, da, day, anomaly) in enumerate(cases):
        tid = str(uuid5(NAMESPACE_OID, f'fixture-{n}'))
        occurred = f'2025-01-{day:02d}T12:00:00+07:00'
        success = n != 5
        tx = {'transaction_id': tid, 'source': {'account_id': src, 'customer_id': 'C'+src},
              'destination': {'account_id': dst, 'type': 'account' if kind == 'TRANSFER' else 'cash' if kind.startswith('CASH') else 'merchant'},
              'amount': amount, 'fee': fee, 'currency': 'VND', 'type': kind, 'channel': 'web', 'status': 'success' if success else 'failed',
              'idempotency_key': tid, 'created_at': occurred, 'posted_at': occurred if success else None,
              'calendar_context': {'tags': ['regular'], 'campaign': False, 'holiday': False}}
        tables['transactions'].append(tx)
        request_type = 'TransferCreated' if kind == 'TRANSFER' else 'TransactionRequested'
        terminal_type = ('TransferCompleted' if kind == 'TRANSFER' else 'TransactionPosted') if success else 'TransferCancelled'
        p = {'sourceAccountId': src, 'destinationAccountId': dst, 'amount': amount, 'fee': fee,
             'currency': 'VND', 'channel': 'web', 'status': 'awaiting_otp' if kind == 'TRANSFER' else 'requested'}
        if kind != 'TRANSFER':
            p['transactionType'] = kind
        result = dict(p, status='success' if success else 'cancelled', sourceBalanceBefore=before, sourceBalanceAfter=after)
        if kind == 'TRANSFER':
            result.update(destinationBalanceBefore=db, destinationBalanceAfter=da)
        for sequence, event_type, payload in [(1, request_type, p), (2, terminal_type, result)]:
            eid = str(uuid5(NAMESPACE_OID, f'{tid}:{event_type}'))
            event = {'event_id': eid, 'aggregate_type': 'transaction', 'aggregate_id': tid, 'event_type': event_type,
                     'schema_version': 2, 'sequence': sequence, 'occurred_at': occurred, 'payload': payload, 'labels': None}
            tables['events'].append(event)
            if kind == 'TRANSFER':
                tables['transfer_events'].append({'eventId': eid, 'transactionId': tid, 'eventType': event_type,
                                                 'schemaVersion': 1, 'sequence': sequence, 'occurredAt': occurred, 'payload': payload})
        tables['labels'].append({'event_id': str(uuid5(NAMESPACE_OID, f'{tid}:{request_type}')), 'transaction_id': tid,
                                'is_anomaly': anomaly, 'scenario_id': 'amount_outlier' if anomaly else 'normal', 'source_profile': 'fixture'})
    tables['events'].sort(key=lambda e: (e['occurred_at'], e['aggregate_type'], e['aggregate_id'], e['sequence']))
    manifest = {'seed': 42, 'schema_version': 2, 'start_at': start, 'end_date_exclusive': '2025-01-21',
                'days': 20, 'source_profile': 'fixture', 'counts': {k: len(v) for k, v in tables.items()}}
    for name, rows in tables.items():
        (root / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root / 'manifest.json').write_text(json.dumps(manifest))


def mutate(root, table, action):
    path = root / f'{table}.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    action(rows)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)

    def codes(self):
        return {i['code'] for i in validate_dataset(self.root)['issues'] if i['severity'] == 'error'}

    def test_valid_all_types_ledger_policy_and_whole_day_preview(self):
        report = validate_dataset(self.root)
        self.assertEqual(report['error_count'], 0, report['issues'])
        self.assertEqual(report['ledger']['final_balances'], {'A': 553, 'B': 496})
        self.assertEqual(report['ledger']['successful_transfer_fees'], 5)
        self.assertEqual(report['ledger']['replayed_transactions'], 6)
        self.assertEqual(len(report['summary']['daily']), 20)
        self.assertEqual(report['summary']['daily']['2025-01-03']['count'], 0)
        self.assertEqual(report['summary']['time_split_preview'][0]['to_exclusive'], '2025-01-11T00:00:00+07:00')
        self.assertEqual(report['feature_policy']['target_scope'], 'all_transaction_types')
        self.assertEqual(len(report['sha256']), 11)

    def test_label_refers_to_terminal(self):
        def change(rows):
            tid = rows[0]['transaction_id']
            rows[0]['event_id'] = str(uuid5(NAMESPACE_OID, f'{tid}:TransferCompleted'))
        mutate(self.root, 'labels', change)
        self.assertIn('label_request', self.codes())

    def test_missing_and_duplicate_labels(self):
        mutate(self.root, 'labels', lambda rows: rows.pop())
        self.assertTrue({'label_join', 'manifest_count'} <= self.codes())
        fixture(self.root)
        mutate(self.root, 'labels', lambda rows: rows.append(rows[0]))
        self.assertIn('duplicate_id', self.codes())

    def test_mutated_balance_and_fee_conservation(self):
        def change(rows):
            e = next(e for e in rows if e['event_type'] == 'TransferCompleted')
            e['payload']['destinationBalanceAfter'] += 1
        mutate(self.root, 'events', change)
        self.assertTrue({'ledger_destination', 'transfer_conservation', 'native_export'} <= self.codes())

    def test_rejected_transfer_changes_no_balances(self):
        def change(rows):
            e = next(e for e in rows if e['event_type'] == 'TransferCancelled')
            e['payload']['sourceBalanceAfter'] -= 100
        mutate(self.root, 'events', change)
        self.assertIn('ledger_after', self.codes())

    def test_future_and_leakage_in_request(self):
        def change(rows):
            e = next(e for e in rows if e['event_type'] == 'TransferCreated')
            e['payload'].update(sourceBalanceAfter=395, is_anomaly=True)
        mutate(self.root, 'events', change)
        self.assertTrue({'schema', 'leakage'} <= self.codes())

    def test_event_order_and_sequence(self):
        mutate(self.root, 'events', lambda rows: rows.reverse())
        self.assertTrue({'event_order', 'sequence', 'lifecycle'} <= self.codes())

    def test_unknown_destination(self):
        mutate(self.root, 'transactions', lambda rows: rows[0]['destination'].update(account_id='unknown'))
        self.assertIn('reference', self.codes())

    def test_boolean_amount_and_naive_timestamp(self):
        mutate(self.root, 'transactions', lambda rows: rows[0].update(amount=True, created_at='2025-01-02T12:00:00'))
        self.assertIn('schema', self.codes())

    def test_future_credit_history(self):
        mutate(self.root, 'customers', lambda rows: rows[0]['credit_profile']['history'][0].update(as_of='2025-02-01'))
        self.assertIn('credit_future', self.codes())

    def test_corrupt_json_and_missing_file(self):
        (self.root / 'labels.jsonl').write_text('{broken\n')
        (self.root / 'accounts.jsonl').unlink()
        self.assertTrue({'jsonl', 'input_file'} <= self.codes())

    def test_native_export_rejects_boolean_schema_version(self):
        mutate(self.root, 'transfer_events', lambda rows: rows[0].update(schemaVersion=True))
        self.assertIn('schema', self.codes())

    def test_bad_event_type_does_not_crash(self):
        mutate(self.root, 'events', lambda rows: rows[0].update(event_type=['invalid']))
        self.assertIn('schema', self.codes())

    def test_cli_fails_and_still_writes_report(self):
        mutate(self.root, 'transactions', lambda rows: rows[0].update(fee=-1))
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_dataset.py'), '--data', str(self.root),
                                    '--output', str(self.root / 'audit')], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(json.loads((self.root / 'audit/validation.json').read_text())['status'], 'failed')
        self.assertTrue((self.root / 'audit/validation.md').exists())


if __name__ == '__main__':
    unittest.main()
