import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fraud_pipeline.validation import validate_dataset
from test_dataset_validation import fixture, mutate


class ContextualValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);fixture(self.root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['scenario_profile'] = 'contextual-v3';manifest['counts']['behavior_profiles'] = 2
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        profiles = [dict(account_id=a, monthly_income=10000000, typical_amount=100,
                         preferred_hours=[10,11,12], preferred_channel='web', payday=5, activity_weight=1.0) for a in ['A','B']]
        (self.root/'behavior_profiles.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in profiles))
        # Existing positive fixture labels have no valid contextual witness.
        mutate(self.root, 'labels', lambda rows: [r.update(scenario_id='contextual_amount') for r in rows if r['is_anomaly']])

    def test_contextual_amount_must_have_unusual_time_and_channel(self):
        report = validate_dataset(self.root)
        self.assertTrue(any(i['code']=='scenario_witness' for i in report['issues']))

    def test_velocity_cannot_use_future_requests_as_witness(self):
        mutate(self.root, 'labels', lambda rows: rows[0].update(is_anomaly=True, scenario_id='velocity_burst'))
        report = validate_dataset(self.root)
        self.assertTrue(any(i['code']=='scenario_witness' and 'Burst' in i['message'] for i in report['issues']))

    def test_graph_fan_in_requires_prior_distinct_sources(self):
        mutate(self.root, 'labels', lambda rows: rows[0].update(is_anomaly=True, scenario_id='graph_fan_in'))
        report = validate_dataset(self.root)
        self.assertTrue(any(i['code']=='scenario_witness' and 'Fan-in' in i['message'] for i in report['issues']))

    def test_missing_profile_is_rejected(self):
        (self.root/'behavior_profiles.jsonl').write_text('')
        report = validate_dataset(self.root)
        self.assertTrue(any(i['code']=='profile_join' for i in report['issues']))

    def configure_v4(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest.update(scenario_profile='contextual-v4', pipeline_config={'stages': [
            {'name': 'transactions', 'options': {'outlier_multipliers': [0.3, 10]}}]})
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        mutate(self.root, 'behavior_profiles', lambda rows: [r.update(preferred_channel='mobile') for r in rows])

    def test_v4_accepts_contextual_amount_in_preferred_hours(self):
        self.configure_v4()
        report = validate_dataset(self.root)
        self.assertEqual(report['error_count'], 0, report['issues'])

    def test_v4_rejects_amount_without_configured_witness(self):
        self.configure_v4()
        mutate(self.root, 'behavior_profiles', lambda rows: [r.update(typical_amount=10000) for r in rows])
        report = validate_dataset(self.root)
        self.assertTrue(any(i['code']=='scenario_witness' for i in report['issues']))

    def test_v4_missing_config_reports_error(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['scenario_profile'] = 'contextual-v4'
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        self.assertTrue(any(i['code']=='scenario_config' for i in validate_dataset(self.root)['issues']))
