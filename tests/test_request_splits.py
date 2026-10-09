"""Phase 2 regression tests: chronological boundaries, joins and provenance."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud_pipeline.config import ROOT, load_config, project_path
from fraud_pipeline.ingestion.request_loader import DatasetError, file_hash, request_samples
from fraud_pipeline.split_export import export_splits, split_intervals
from fraud_pipeline.validation import validate_dataset
from test_dataset_validation import fixture, mutate

RATIOS = {'train': 0.5, 'val': 0.3, 'test': 0.2, 'time_based': True}


class RequestSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        fixture(self.source)
        self.report = self.root / 'validation.json'
        self.output = self.root / 'splits'
        self.audit()

    def audit(self):
        self.report.write_text(json.dumps(validate_dataset(self.source)))

    def export(self, output=None, **kwargs):
        return export_splits(self.source, self.report, output or self.output, RATIOS, **kwargs)

    def test_all_types_join_excludes_terminal_and_trace_is_resolvable(self):
        result = self.export()
        events = (self.source / 'events.jsonl').read_text().splitlines()
        types, ids = set(), set()
        for name, stat in result['splits'].items():
            path = self.output / stat['file']
            self.assertEqual(file_hash(path), stat['sha256'])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), stat['count'])
            for row in rows:
                self.assertNotIn(row['transaction_id'], ids)
                ids.add(row['transaction_id'])
                types.add(row['type'])
                source = json.loads(events[row['source_event_line'] - 1])
                self.assertEqual(source['event_id'], row['event_id'])
                self.assertEqual(source['aggregate_id'], row['transaction_id'])
                self.assertEqual(source['payload']['amount'], row['amount'])
                self.assertNotIn('status', row)
                self.assertNotIn('sourceBalanceAfter', row)
                self.assertNotIn('scenario_id', row)
                self.assertNotIn('risk', row)
                self.assertIn(row['label'], (0, 1))
                self.assertEqual(set(row), set(result['sample_fields']))
        self.assertEqual(len(ids), 6)
        self.assertEqual(types, {'TRANSFER', 'CASH_IN', 'CASH_OUT', 'PAYMENT', 'DEBIT'})
        self.assertEqual([result['splits'][s]['count'] for s in ('train', 'validation', 'test')], [3, 1, 2])
        self.assertTrue(result['warnings'])  # No resampling when validation has only one class.

    def test_reproducible_bytes_and_config_paths_ignore_working_directory(self):
        self.export()
        second = self.root / 'second'
        self.export(second)
        for file in self.output.iterdir():
            self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(project_path('configs/dataset.json'), ROOT / 'configs/dataset.json')
        self.assertEqual(load_config('configs/dataset.json')['split'], RATIOS)

    def test_boundary_equal_timestamps_go_to_same_set(self):
        # Put first two requests and their terminal events at the train cutoff.
        moment = '2025-01-11T00:00:00+07:00'
        def change_tx(rows):
            for row in rows[:2]:
                row.update(created_at=moment, posted_at=moment)
        mutate(self.source, 'transactions', change_tx)
        tx_ids = [json.loads(s)['transaction_id'] for s in (self.source / 'transactions.jsonl').read_text().splitlines()[:2]]
        for table in ['events', 'transfer_events']:
            def change(rows):
                for row in rows:
                    if row.get('aggregate_id', row.get('transactionId')) in tx_ids:
                        row['occurred_at' if table == 'events' else 'occurredAt'] = moment
                # Preserve the hand-built ledger transaction order within ties.
                rows.sort(key=lambda r: r.get('occurred_at', r.get('occurredAt')))
            mutate(self.source, table, change)
        # Loader itself must handle same-time requests without row-count slicing.
        rows = list(request_samples(self.source, split_intervals(json.loads((self.source/'manifest.json').read_text()), RATIOS)['train'][0].tzinfo))
        intervals = split_intervals(json.loads((self.source/'manifest.json').read_text()), RATIOS)
        for row in rows:
            if row['transaction_id'] in tx_ids:
                from fraud_pipeline.validation.core import timestamp
                self.assertTrue(intervals['validation'][0] <= timestamp(row['timestamp']) < intervals['validation'][1])

    def test_stale_validation_denies_export(self):
        mutate(self.source, 'labels', lambda rows: rows[0].update(is_anomaly=True))
        with self.assertRaisesRegex(DatasetError, 'Stale validation'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_failed_audit_denies_export(self):
        report = json.loads(self.report.read_text())
        report.update(status='failed', error_count=1)
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(DatasetError, 'validation failed'):
            self.export()

    def test_report_from_another_directory_denied(self):
        report = json.loads(self.report.read_text())
        report['dataset'] = str(self.root / 'different')
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(DatasetError, 'different dataset'):
            self.export()

    def test_missing_label_and_duplicate_request_fail_loader(self):
        mutate(self.source, 'labels', lambda rows: rows.pop())
        from datetime import timezone
        with self.assertRaisesRegex(DatasetError, 'missing label'):
            list(request_samples(self.source, timezone.utc))
        fixture(self.source)
        def change(rows):
            i = next(i for i, e in enumerate(rows) if e['event_type'] == 'TransferCreated')
            rows.insert(i + 1, rows[i])
        mutate(self.source, 'events', change)
        with self.assertRaisesRegex(DatasetError, 'duplicate request'):
            list(request_samples(self.source, timezone.utc))

    def test_label_must_point_to_request_and_extra_label_denied(self):
        def change(rows):
            rows[0]['event_id'] = 'wrong-event'
        mutate(self.source, 'labels', change)
        from datetime import timezone
        with self.assertRaisesRegex(DatasetError, 'label must point'):
            list(request_samples(self.source, timezone.utc))
        fixture(self.source)
        mutate(self.source, 'labels', lambda rows: rows.append(dict(rows[0], event_id='extra-event', transaction_id='extra-tx')))
        with self.assertRaisesRegex(DatasetError, 'no request event'):
            list(request_samples(self.source, timezone.utc))

    def test_overwrite_preserves_previous_export_and_failed_run_does_not_replace(self):
        self.export()
        old = (self.output / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(DatasetError, 'already exists'):
            self.export()
        self.export(overwrite=True)
        backups = list(self.root.glob('splits.previous-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'manifest.json').read_bytes(), old)
        mutate(self.source, 'labels', lambda rows: rows.pop())
        with self.assertRaises(DatasetError):
            self.export(overwrite=True)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), old)

    def test_source_mutation_during_export_prevents_promotion(self):
        original = request_samples
        def changed(source, timezone):
            yield from original(source, timezone)
            with (source / 'accounts.jsonl').open('a') as stream:
                stream.write('\n')
        with patch('fraud_pipeline.split_export.request_samples', changed):
            with self.assertRaisesRegex(DatasetError, 'Source changed during export'):
                self.export()
        self.assertFalse(self.output.exists())

    def test_invalid_ratios_and_unsafe_output_denied(self):
        manifest = json.loads((self.source / 'manifest.json').read_text())
        for ratios in [dict(RATIOS, train=0.7), dict(RATIOS, val=False), dict(RATIOS, time_based=False),
                       dict(RATIOS, train='NaN'), dict(RATIOS, train=0.01, val=0.79)]:
            with self.subTest(ratios=ratios), self.assertRaises(DatasetError):
                split_intervals(manifest, ratios)
        for output in [self.source, self.source/'nested', self.root]:
            with self.subTest(output=output), self.assertRaises(DatasetError):
                self.export(output)

    def test_cli_runs_from_outside_repo(self):
        completed = subprocess.run([sys.executable, str(ROOT/'scripts/run_pipeline.py'), '--stage', 'split',
                                    '--data', str(self.source), '--validation-report', str(self.report),
                                    '--output', str(self.output)], cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('train: 3 samples', completed.stdout)
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['total_samples'], 6)


if __name__ == '__main__':
    unittest.main()
