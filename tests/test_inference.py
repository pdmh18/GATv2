import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from fraud_pipeline.features.causal import REQUESTS, microseconds
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.features.export import export_features
from fraud_pipeline.inference.contracts import normalize_event
from fraud_pipeline.inference.history import EventStoreHistory, JsonlHistory, MongoHistory
from fraud_pipeline.inference.predictor import ReplayPredictor
from fraud_pipeline.inference.stream_inference import consume, handle_message
from fraud_pipeline.ingestion.request_loader import file_hash
from fraud_pipeline.models.batching import TemporalBatches
from fraud_pipeline.models.variants import build_model
from fraud_pipeline.registry.bundle import export_bundle
from fraud_pipeline.registry.save_load import save
from fraud_pipeline.split_export import export_splits
from fraud_pipeline.validation import validate_dataset
from test_dataset_validation import fixture, mutate

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        raw = cls.root/'raw'
        raw.mkdir()
        fixture(raw)

        def enrich(rows):
            for row in rows:
                if row['event_type'] in REQUESTS:
                    row['payload'].update(device_id='D1', platform='web',
                                          authentication_method='OTP', authentication_verified=False)
        mutate(raw, 'events', enrich)
        def enrich_native(rows):
            for row in rows:
                if row['eventType'] == 'TransferCreated':
                    row['payload'].update(device_id='D1', platform='web',
                                          authentication_method='OTP', authentication_verified=False)
        mutate(raw, 'transfer_events', enrich_native)
        config = json.loads((ROOT/'configs/features-v3.json').read_text())
        report = cls.root/'validation.json'
        audit = validate_dataset(raw)
        assert audit['error_count'] == 0, audit['issues']
        report.write_text(json.dumps(audit))
        export_splits(raw, report, cls.root/'splits', config['split'])
        export_features(raw, report, cls.root/'splits', cls.root/'features', config['features'])
        cls.data = TemporalDataset(cls.root/'features')
        cls.events = [json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
        cls.requests = [e for e in cls.events if e['event_type'] in REQUESTS]
        cls.training = cls.root/'training'
        cls.training.mkdir()
        for name in ('manifest.json', 'preprocessing.json', 'node_preprocessing.json', 'nodes.json'):
            (cls.training/('feature-'+name)).write_bytes((cls.root/'features'/name).read_bytes())
        model_config = dict(kind='gatv2', in_dim=12, edge_dim=len(cls.data.manifest['feature_names']),
                            hidden_dim=8, heads=2, num_layers=2, dropout=0., masked_feature_indices=[0])
        torch.manual_seed(42)
        cls.model = build_model(model_config).eval()
        metadata = {'feature_manifest_sha256': file_hash(cls.root/'features/manifest.json'), 'snapshot_seconds': 3600}
        save(cls.model, cls.training/'best.pt', model_config=model_config, metadata=metadata)
        cls.selection = cls.root/'selection.json'
        cls.selection.write_text(json.dumps({'status': 'frozen_before_test', 'selected_model': 'gatv2',
            'feature_manifest_sha256': metadata['feature_manifest_sha256'],
            'models': {'gatv2': {'sha256': file_hash(cls.training/'best.pt'), 'threshold': .7}}}))
        export_bundle(cls.training, cls.selection, cls.root/'bundle', model_version='gatv2-test')
        cls.predictor = ReplayPredictor(cls.root/'bundle')

    def assert_context_equal(self, actual, expected):
        for name in ('x', 'edge_index', 'target_edge_index', 'edge_features'):
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6, err_msg=name)

    def test_replay_matches_offline_tensors_and_gatv2_scores_all_business_types(self):
        for i, request in enumerate(self.requests):
            with self.subTest(i=i):
                expected = self.data.context(i)
                actual = self.predictor.inputs(request, self.events)
                self.assert_context_equal(actual, expected)
                tensors = {k: torch.from_numpy(expected[k]).to(dtype=torch.float32 if k in ('x', 'edge_features') else torch.long)
                           for k in ('x', 'edge_index', 'target_edge_index', 'edge_features')}
                with torch.inference_mode():
                    score = float(torch.sigmoid(self.model(**tensors)).item())
                response = self.predictor.predict(request, self.events)
                self.assertAlmostEqual(response['risk_score'], score, places=7)
                self.assertEqual(response['predicted_label'], int(score >= .7))
                self.assertEqual(set(response), {'transaction_id', 'risk_score', 'predicted_label', 'threshold', 'model_version', 'feature_version'})

    def test_snapshot_matches_training_bucket_and_transaction_features_stay_current(self):
        # A wide bucket deliberately contains several requests; graph must use its first cutoff.
        predictor = copy.copy(self.predictor)
        predictor.bundle = dict(predictor.bundle, snapshot_seconds=60*86400)
        with tempfile.TemporaryDirectory() as folder:
            batches = TemporalBatches(self.data, 60*86400, 2, folder, allow_test=True)
            for batch in batches.iterate('train'):
                for position, index in enumerate(batch['indices']):
                    actual = predictor.inputs(self.requests[index], self.events)
                    self.assert_context_equal(actual, dict(batch,
                        target_edge_index=batch['target_edge_index'][:, position:position+1],
                        edge_features=batch['edge_features'][position:position+1]))

    def test_future_same_timestamp_and_duplicate_delivery_do_not_change_score(self):
        request = self.requests[3]
        at = microseconds(request['occurred_at'])
        prefix = [e for e in self.events if microseconds(e['occurred_at']) < at]
        before = self.predictor.predict(request, prefix)
        self.assertEqual(before, self.predictor.predict(request, self.events))
        duplicate = prefix + [copy.deepcopy(prefix[-1])]
        self.assertEqual(before, self.predictor.predict(request, duplicate))
        conflict = copy.deepcopy(prefix[-1])
        conflict['payload']['amount'] += 1
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.predictor.inputs(request, prefix+[conflict])
        with self.assertRaisesRegex(ValueError, 'chronological'):
            self.predictor.inputs(request, list(reversed(prefix)))

    def test_raw_review_contract_and_native_transfer_envelope(self):
        raw = {'transaction_id': 'TX001', 'source_account': 'A', 'destination_account': 'B',
               'amount': 50000000, 'transaction_type': 'TRANSFER', 'channel': 'MOBILE',
               'authentication_method': 'OTP', 'authentication_verified': True,
               'device_id': 'DEVICE01', 'platform': 'ANDROID', 'created_at': '2025-01-19T00:50:00+07:00'}
        response = handle_message(self.predictor, JsonlHistory(self.root/'raw/events.jsonl'), raw)
        self.assertEqual(response['transaction_id'], 'TX001')
        self.assertTrue(0 <= response['risk_score'] <= 1)
        native = json.loads((self.root/'raw/transfer_events.jsonl').read_text().splitlines()[0])
        canonical = next(e for e in self.requests if e['aggregate_id'] == native['transactionId'])
        native['payload'] = canonical['payload']
        self.assertEqual(self.predictor.predict(native, self.events), self.predictor.predict(canonical, self.events))
        for changes in ({'posted_at': raw['created_at']}, {'risk_score': .9}, {'authentication_verified': 'true'}):
            with self.assertRaises(ValueError):
                normalize_event(dict(raw, **changes))

    def test_bundle_rejects_wrong_checkpoint_and_modified_policy(self):
        selection = json.loads(self.selection.read_text())
        selection['models']['gatv2']['sha256'] = 'wrong'
        bad = self.root/'bad-selection.json'
        bad.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, 'Checkpoint differs'):
            export_bundle(self.training, bad, self.root/'bad-bundle', model_version='bad')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'bundle'
            export_bundle(self.training, self.selection, target, model_version='test')
            policy = json.loads((target/'bundle.json').read_text())
            policy['threshold'] = .1
            (target/'bundle.json').write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, 'Serving policy differs'):
                ReplayPredictor(target)
            policy['threshold'] = .7
            (target/'bundle.json').write_text(json.dumps(policy))
            with (target/'preprocessing.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ReplayPredictor(target)

    def test_native_bank_records_extract_only_event_and_ignore_cancel_without_balances(self):
        native = json.loads((self.root/'raw/transfer_events.jsonl').read_text().splitlines()[0])
        record = {'id': native['eventId'], 'type': native['eventType'], 'at': native['occurredAt'],
                  'event': native, 'transfer': {'posted_at': '2099-01-01T00:00:00Z',
                                               'authentication': {'verified': True}, 'risk_score': 1.}}
        self.assertEqual(normalize_event(record), normalize_event(native))
        cancelled = dict(native, eventType='TransferCancelled', sequence=2,
                         payload=dict(native['payload'], status='cancelled'))
        cancelled['payload'] = {k: v for k, v in cancelled['payload'].items()
                                if k not in {'device_id', 'platform', 'authentication_method', 'authentication_verified'}}
        self.assertIsNone(normalize_event(cancelled))
        non_transaction = {'id': 'account-event', 'type': 'BankAccountCreated',
                           'at': native['occurredAt'], 'account': {'private_profile': 'excluded'}}
        self.assertIsNone(normalize_event(non_transaction))
        client = Mock()
        client.get_stream.return_value = [SimpleNamespace(data=json.dumps(e).encode()) for e in [non_transaction, record]]
        prefix = EventStoreHistory(client, 'banking').read_before(microseconds(native['occurredAt'])+1)
        self.assertEqual(prefix, [native])
        bad = dict(record, id='different')
        with self.assertRaisesRegex(ValueError, 'metadata disagree'):
            normalize_event(bad)

    def objectid_request(self, request):
        from uuid import NAMESPACE_OID, uuid5
        result = copy.deepcopy(request)
        transaction_id = str(uuid5(NAMESPACE_OID, 'runtime-account-test:' + result['aggregate_id']))
        result['aggregate_id'] = transaction_id
        result['event_id'] = str(uuid5(NAMESPACE_OID, transaction_id + ':' + result['event_type']))
        result['payload']['sourceAccountId'] = '6623a1f89c30111111111111'
        result['payload']['destinationAccountId'] = '6623a1f89c30222222222222'
        return result

    def test_unseen_objectids_have_distinct_cold_start_nodes_and_real_inference(self):
        request = self.objectid_request(self.requests[0])
        frozen = copy.deepcopy(self.predictor.nodes)
        context = self.predictor.inputs(request, [])
        source, destination = context['target_edge_index'][:, 0]
        self.assertNotEqual(source, 0)
        self.assertNotEqual(destination, 0)
        self.assertNotEqual(source, destination)
        self.assertEqual(context['x'].shape[0], len(frozen) + 2)
        self.assertEqual(context['x'].shape[1], len(self.predictor.manifest['node_feature_names']))
        response = self.predictor.predict(request, [])
        self.assertEqual(response['transaction_id'], request['aggregate_id'])
        self.assertTrue(0 <= response['risk_score'] <= 1)
        self.assertEqual(self.predictor.nodes, frozen)
        self.assertEqual(response, ReplayPredictor(self.root/'bundle').predict(request, []))

    def test_account_rename_preserves_graph_features_and_score(self):
        rename = {'A': '6623a1f89c30111111111111', 'B': '6623a1f89c30222222222222'}
        history = copy.deepcopy(self.events)
        for event in history:
            for field in ('sourceAccountId', 'destinationAccountId'):
                identity = event['payload'].get(field)
                if identity in rename:
                    event['payload'][field] = rename[identity]
        for request in self.requests:
            if request['event_type'] != 'TransferCreated':
                continue
            mapped = copy.deepcopy(request)
            for field in ('sourceAccountId', 'destinationAccountId'):
                mapped['payload'][field] = rename[mapped['payload'][field]]
            before = self.predictor.inputs(request, self.events)
            after = self.predictor.inputs(mapped, history)
            np.testing.assert_allclose(before['edge_features'], after['edge_features'])
            for old, new in zip(before['target_edge_index'][:, 0], after['target_edge_index'][:, 0]):
                np.testing.assert_allclose(before['x'][old], after['x'][new])
            self.assertNotEqual(*after['target_edge_index'][:, 0])
            self.assertAlmostEqual(self.predictor.predict(request, self.events)['risk_score'],
                                   self.predictor.predict(mapped, history)['risk_score'], places=7)

    def test_new_accounts_in_existing_snapshot_keep_identity_without_future_history(self):
        predictor = copy.copy(self.predictor)
        predictor.bundle = dict(predictor.bundle, snapshot_seconds=100000000000)
        request = self.objectid_request(self.requests[-1])
        context = predictor.inputs(request, self.events)
        self.assertTrue(np.all(context['target_edge_index'] > 0))
        self.assertNotEqual(*context['target_edge_index'][:, 0])
        future = self.objectid_request(self.requests[-1])
        future['payload']['sourceAccountId'] = '6623a1f89c30333333333333'
        future['occurred_at'] = '2099-01-01T00:00:00Z'
        with_future = predictor.inputs(request, [*self.events, future])
        self.assert_context_equal(context, with_future)
        self.assertEqual(self.predictor.predict(request, self.events),
                         self.predictor.predict(request, [*self.events, future]))

    def test_pipeline_inference_uses_configured_bundle_and_history(self):
        from fraud_pipeline.pipeline import stage_infer
        with patch('fraud_pipeline.inference.stream_inference.run') as worker:
            stage_infer(str(ROOT/'configs/inference.json'))
        config = worker.call_args.args[0]
        self.assertTrue(Path(config['bundle']).is_absolute())
        self.assertTrue(Path(config['history']['path']).is_absolute())

    def test_history_adapters_strict_cutoff_and_same_history_after_restart(self):
        at = microseconds(self.requests[3]['occurred_at'])
        expected = JsonlHistory(self.root/'raw/events.jsonl').read_before(at)
        client = Mock()
        client.get_stream.return_value = [SimpleNamespace(data=json.dumps(e).encode()) for e in reversed(self.events)]
        actual = EventStoreHistory(client, 'banking').read_before(at)
        self.assertEqual(sorted(actual, key=lambda e: e['event_id']), sorted(expected, key=lambda e: e['event_id']))
        self.assertEqual(self.predictor.predict(self.requests[3], actual), self.predictor.predict(self.requests[3], expected))
        collection = Mock()
        collection.find.return_value = [{'timestamp_us': microseconds(e['occurred_at']), 'event': e} for e in expected]
        self.assertEqual(MongoHistory(collection).read_before(at), expected)
        collection.find.assert_called_once_with({'timestamp_us': {'$lt': at}}, {'_id': 0})
        restarted = ReplayPredictor(self.root/'bundle')
        self.assertEqual(restarted.predict(self.requests[3], expected), self.predictor.predict(self.requests[3], expected))

    def test_kafka_commits_only_after_ack_and_skips_terminal_predictions(self):
        history = JsonlHistory(self.root/'raw/events.jsonl')
        consumer = Mock()
        consumer.__iter__ = Mock(return_value=iter([SimpleNamespace(value=self.requests[3])]))
        producer = Mock()
        producer.send.return_value.get.side_effect = TimeoutError('no acknowledgement')
        with self.assertRaises(TimeoutError):
            consume(consumer, producer, self.predictor, history, output_topic='risk')
        consumer.commit.assert_not_called()
        consumer.__iter__ = Mock(return_value=iter([SimpleNamespace(value=self.requests[3])]))
        producer.send.return_value.get.side_effect = None
        consume(consumer, producer, self.predictor, history, output_topic='risk')
        consumer.commit.assert_called_once()
        terminal = next(e for e in self.events if e['event_type'] == 'TransferCompleted')
        self.assertIsNone(handle_message(self.predictor, history, terminal))


if __name__ == '__main__':
    unittest.main()
