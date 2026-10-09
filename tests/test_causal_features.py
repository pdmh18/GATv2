import copy
import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fraud_pipeline.features.causal import State, names, replay
from fraud_pipeline.features.export import export_features
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.split_export import export_splits
from fraud_pipeline.validation import validate_dataset
from test_dataset_validation import fixture, mutate

OPTIONS = json.loads((ROOT / 'configs/dataset.json').read_text())['features']


def event(second, kind='TransferCreated', src='A', dst='B', amount=100):
    at = datetime.fromisoformat('2025-01-01T12:00:00+07:00') + timedelta(seconds=second)
    return {'aggregate_type': 'transaction', 'aggregate_id': f'{src}-{second}', 'event_type': kind,
            'occurred_at': at.isoformat(), 'payload': {'sourceAccountId': src, 'destinationAccountId': dst,
            'amount': amount, 'fee': 0, 'channel': 'web'}}


class ReplayTests(unittest.TestCase):
    def values(self, events):
        return [dict(zip(names(OPTIONS), values)) for _, values, _ in replay(events, State(OPTIONS))]

    def test_current_terminal_and_same_timestamp_requests_are_excluded(self):
        rows = self.values([event(0), event(0, 'TransferCompleted'), event(0, src='A', dst='C'), event(1)])
        for row in rows[:2]:
            self.assertEqual(row['log_src_history_count'], 0)
            self.assertEqual(row['has_type_completed_history'], 0)
        self.assertAlmostEqual(rows[2]['log_src_history_count'], math.log1p(2))
        self.assertAlmostEqual(rows[2]['log_src_type_completed_median_amount'], math.log1p(100))

    def test_failed_request_contributes_to_velocity_but_not_completed_mean(self):
        rows = self.values([event(0), event(0, 'TransferCancelled'), event(1)])
        self.assertAlmostEqual(rows[-1]['log_src_request_count_120s'], math.log1p(1))
        self.assertEqual(rows[-1]['has_type_completed_history'], 0)
        self.assertEqual(rows[-1]['new_receiver'], 0)

    def test_future_and_label_changes_do_not_change_past_features(self):
        events = [event(0), event(0, 'TransferCompleted'), event(1)]
        before = self.values(events)
        changed = copy.deepcopy(events)
        for e in changed:
            e['labels'] = {'is_anomaly': True, 'scenario': 'oracle'}
            e['payload']['sourceBalanceAfter'] = 999999
        self.assertEqual(before, self.values(changed + [event(99999, amount=999999)] )[:len(before)])

    def test_window_boundary_is_inclusive_and_old_requests_expire(self):
        rows = self.values([event(0), event(120), event(121)])
        self.assertAlmostEqual(rows[1]['log_src_request_count_120s'], math.log1p(1))
        self.assertAlmostEqual(rows[2]['log_src_request_count_120s'], math.log1p(1))

    def test_backward_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chronological'):
            self.values([event(1), event(0)])

    def test_equivalent_timezones_have_identical_calendar_features(self):
        original = event(0)
        utc = copy.deepcopy(original)
        utc['occurred_at'] = '2025-01-01T05:00:00+00:00'
        zone = datetime.fromisoformat(original['occurred_at']).tzinfo
        self.assertEqual(State(OPTIONS, zone).score(original), State(OPTIONS, zone).score(utc))

    def test_personal_median_ignores_other_type_and_large_outlier(self):
        rows = []
        for i, amount in enumerate((10,10,10,10,1000)):
            rows.extend([event(i*2,amount=amount),event(i*2,'TransferCompleted',amount=amount)])
        income = event(11,'TransactionPosted',amount=100000)
        income['payload']['transactionType'] = 'CASH_IN'
        rows.extend([income,event(12,amount=100)])
        values = self.values(rows)[-1]
        self.assertAlmostEqual(values['log_src_type_completed_median_amount'],math.log1p(10))
        self.assertAlmostEqual(values['log_src_type_completed_count'],math.log1p(5))
        self.assertEqual(values['type_amount_reference_reliable'],1)
        self.assertAlmostEqual(values['log_amount_to_type_median'],math.log1p(10))

    def test_unreliable_reference_does_not_emit_amount_ratio(self):
        rows = self.values([event(0),event(0,'TransferCompleted'),event(1,amount=10000)])
        self.assertEqual(rows[-1]['type_amount_reference_reliable'],0)
        self.assertEqual(rows[-1]['log_amount_to_type_median'],0)

    def test_neighborhood_window_remains_short_when_graph_window_is_long(self):
        rows = self.values([event(0,src='B',dst='A'),event(0,'TransferCompleted',src='B',dst='A'),event(601)])
        self.assertEqual(rows[-1]['log_src_transfer_attempt_distinct_sources'],0)


class TensorTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy required for tensor export tests')
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw'
        self.source.mkdir()
        fixture(self.source)
        self.report = self.root / 'validation.json'
        self.report.write_text(json.dumps(validate_dataset(self.source)))
        self.splits = self.root / 'splits'
        export_splits(self.source, self.report, self.splits, {'train': .5, 'val': .3, 'test': .2, 'time_based': True})
        self.output = self.root / 'features'
        self.options = dict(OPTIONS, graph_window_seconds=604800, node_history_window_seconds=604800, bidirectional_graph=False)

    def build(self):
        return export_features(self.source, self.report, self.splits, self.output, self.options)

    def test_shapes_cash_in_direction_masks_and_causal_graph(self):
        result = self.build()
        data = TemporalDataset(self.output)
        self.assertEqual(result['request_count'], 6)
        self.assertEqual(data.arrays['transaction_features'].shape, (6, len(names(OPTIONS))))
        self.assertEqual(data.context(0)['edge_index'].shape, (2, 0))
        self.assertNotIn('label', data.context(1))
        self.assertTrue(self.np.array_equal(data.arrays['flow_edge_index'][:, 1], data.arrays['target_edge_index'][::-1, 1]))
        self.assertTrue(self.np.array_equal(data.context(1)['edge_index'], data.arrays['target_edge_index'][:, :1]))
        cash_edge = tuple(data.arrays['flow_edge_index'][:, 1])
        self.assertIn(cash_edge, {tuple(edge) for edge in data.context(2)['edge_index'].T})
        self.assertTrue(self.np.all(sum(data.arrays[n + '_mask'].astype(int) for n in ('train', 'validation', 'test')) == 1))
        for i in range(len(data)):
            context = data.context(i)
            start, end = context['graph_interval']
            self.assertTrue(self.np.all(data.arrays['graph_timestamps_us'][start:end] < context['timestamp_us']))

    def test_scaler_uses_train_only(self):
        self.build()
        data = TemporalDataset(self.output)
        prep = json.loads((self.output / 'preprocessing.json').read_text())
        self.assertEqual(prep['fit_count'], 3)
        train = data.arrays['transaction_features'][data.arrays['train_mask']]
        self.assertTrue(self.np.allclose(train[:, prep['numeric_indices']].mean(axis=0), 0, atol=1e-6))
        # Held-out transfer of 1000 must not enter the learned amount mean.
        index = names(OPTIONS).index('log_amount')
        self.assertAlmostEqual(prep['mean'][prep['numeric_indices'].index(index)],
                               self.np.mean(self.np.log1p([100, 200, 70])), places=6)

    def test_split_corruption_is_rejected_without_publishing_output(self):
        with (self.splits / 'test.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.build()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.build()
        self.assertEqual(len(TemporalDataset(self.output)), 6)

    def test_heldout_amount_and_labels_cannot_change_training_context(self):
        self.build()
        before = TemporalDataset(self.output)
        train_features = before.arrays['transaction_features'][:3].copy()
        train_graphs = [before.context(i)['edge_index'].copy() for i in range(3)]
        train_nodes = [before.context(i)['x'].copy() for i in range(3)]
        prep = json.loads((self.output / 'preprocessing.json').read_text())
        mutate(self.source, 'transactions', lambda rows: rows[-1].update(amount=1000000))
        tid = json.loads((self.source / 'transactions.jsonl').read_text().splitlines()[-1])['transaction_id']
        mutate(self.source, 'events', lambda rows: [e['payload'].update(amount=1000000) for e in rows if e['aggregate_id'] == tid])
        mutate(self.source, 'transfer_events', lambda rows: [e['payload'].update(amount=1000000) for e in rows if e['transactionId'] == tid])
        mutate(self.source, 'labels', lambda rows: [r.update(is_anomaly=not r['is_anomaly'],
            scenario_id='normal' if r['is_anomaly'] else 'amount_outlier') for r in rows])
        report = validate_dataset(self.source)
        self.assertEqual(report['error_count'], 0, report['issues'])
        self.report.write_text(json.dumps(report))
        export_splits(self.source, self.report, self.splits, {'train': .5, 'val': .3, 'test': .2, 'time_based': True}, overwrite=True)
        alternate = self.root / 'alternate-features'
        export_features(self.source, self.report, self.splits, alternate, self.options)
        after = TemporalDataset(alternate)
        self.assertTrue(self.np.array_equal(train_features, after.arrays['transaction_features'][:3]))
        self.assertEqual(prep, json.loads((alternate / 'preprocessing.json').read_text()))
        for i, graph in enumerate(train_graphs):
            self.assertTrue(self.np.array_equal(graph, after.context(i)['edge_index']))
            self.assertTrue(self.np.array_equal(train_nodes[i], after.context(i)['x']))

    def test_dynamic_nodes_differ_and_expire_without_new_node_events(self):
        self.build()
        data = TemporalDataset(self.output)
        node_ids = {n['id']: i for i,n in enumerate(json.loads((self.output/'nodes.json').read_text()))}
        x = data.context(1)['x']
        self.assertEqual(x.shape[1],len(data.manifest['node_feature_names']))
        self.assertFalse(self.np.array_equal(x[node_ids['A']],x[node_ids['B']]))
        # Day 17 is >7 days after B's last settled activity on day 8.
        x = data.context(4)['x']
        self.assertTrue(self.np.array_equal(x[node_ids['B'],-2:],[0,0]))


if __name__ == '__main__':
    unittest.main()
