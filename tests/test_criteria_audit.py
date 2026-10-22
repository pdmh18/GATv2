import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fraud_pipeline.evaluation.criteria_audit import load_catalog, readiness_rows, render_catalog, summarize_events

ROOT = Path(__file__).resolve().parents[1]


class CriteriaAuditTests(unittest.TestCase):
    def test_all_49_ids_are_unique_and_document_is_generated_from_catalog(self):
        catalog = load_catalog(ROOT/'specs/anomaly_criteria.v2.json')
        self.assertEqual(len(catalog['criteria']), 49)
        document = (ROOT/'specs/Bo_tieu_chi_phat_hien_giao_dich_bat_thuong.md').read_text()
        body = document.split('<!-- CRITERIA_CATALOG_START -->')[1].split('<!-- CRITERIA_CATALOG_END -->')[0]
        self.assertEqual(body.strip(), render_catalog(catalog).strip())
        self.assertTrue(all(row['normal_counterexample'] for row in catalog['criteria']))

    def test_catalog_rejects_duplicate_id_or_missing_counterexample(self):
        original = json.loads((ROOT/'specs/anomaly_criteria.v2.json').read_text())
        for corruption in ('id', 'normal_counterexample'):
            catalog = copy.deepcopy(original)
            catalog['criteria'][0][corruption] = catalog['criteria'][1]['id'] if corruption == 'id' else ''
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'catalog.json'
                path.write_text(json.dumps(catalog))
                with self.assertRaises(ValueError):
                    load_catalog(path)

    def test_observable_fields_distinguish_false_from_missing_and_devices_can_be_shared(self):
        events = [dict(event_type='TransferCreated', payload=dict(sourceAccountId='A', device_id='D1',
                           platform='WEB', authentication_verified=False)),
                  dict(event_type='TransferCreated', payload=dict(sourceAccountId='B', device_id='D1')),
                  dict(event_type='TransferCompleted', payload=dict(authentication_verified=True)),
                  dict(event_type='CustomerProfileCreated', payload=dict(monthly_income=10000000))]
        capabilities, report = summarize_events(events)
        self.assertEqual(report['request_field_coverage']['authentication_verified']['present'], 1)
        self.assertEqual(report['request_field_coverage']['authentication_verified']['fraction'], .5)
        self.assertEqual(report['devices_shared_across_accounts'], 1)
        self.assertEqual(report['max_accounts_per_device'], 2)
        self.assertNotIn('observed_profile', capabilities)
        self.assertNotIn('watchlist_asof', capabilities)
        self.assertNotIn('login_history', capabilities)

    def test_readiness_is_not_matching_and_calendar_support_does_not_leak_future(self):
        catalog = load_catalog(ROOT/'specs/anomaly_criteria.v2.json')
        day = 86400*1000000
        times = np.asarray([0, 89*day, 90*day, 181*day])
        masks = dict(train=np.asarray([1, 1, 0, 0], dtype=bool),
                     validation=np.asarray([0, 0, 1, 0], dtype=bool), test=np.asarray([0, 0, 0, 1], dtype=bool))
        rows = {r['id']: r for r in readiness_rows(catalog, {'request_history', 'settled_history'}, times, masks, 0)}
        self.assertEqual(rows['A01']['source_readiness'], 'sources_present')
        self.assertEqual(rows['A01']['train_calendar_upper_bound'], 0)
        self.assertEqual(rows['A01']['validation_calendar_upper_bound'], 1)
        self.assertEqual(rows['G01']['source_readiness'], 'missing_sources')
        self.assertEqual(rows['F02']['source_readiness'], 'partial_sources')
        self.assertEqual(rows['A01']['criterion_matching'], 'not_implemented_by_this_eda')
        self.assertNotIn('matched', rows['A01'])
        self.assertEqual(rows['A04']['test_calendar_upper_bound'], 0)
        self.assertIn('minimum_amount', rows['A02']['unconfigured_parameters'])


if __name__ == '__main__':
    unittest.main()
