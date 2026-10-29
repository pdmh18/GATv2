import json
import math
import unittest
from pathlib import Path

from fraud_pipeline.inference.response import (KEY_FIELDS, RESPONSE_FIELDS, build_response,
                                               response_key, validate_response)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT/'specs/model_response.v1.json').read_text())


class ResponseContractTests(unittest.TestCase):
    def test_published_schema_matches_runtime_contract(self):
        self.assertEqual(set(SCHEMA['required']), set(RESPONSE_FIELDS))
        self.assertEqual(set(SCHEMA['properties']), set(RESPONSE_FIELDS))
        self.assertFalse(SCHEMA['additionalProperties'])
        self.assertEqual(SCHEMA['properties']['predicted_label']['enum'], [0, 1])
        for name in ('risk_score', 'threshold'):
            self.assertEqual(SCHEMA['properties'][name]['minimum'], 0)
            self.assertEqual(SCHEMA['properties'][name]['maximum'], 1)
        for example in SCHEMA['examples']:
            self.assertEqual(validate_response(example), example)

    def test_build_response_uses_threshold_and_exact_fields(self):
        response = build_response(transaction_id='TX001', risk_score=.9821, threshold=.95,
                                  model_version='gatv2-test', feature_version='schema-3-test')
        self.assertEqual(tuple(response), RESPONSE_FIELDS)
        self.assertEqual(response['predicted_label'], 1)
        self.assertIs(type(response['predicted_label']), int)
        self.assertIs(type(response['risk_score']), float)
        self.assertEqual(response_key(response), 'TX001:gatv2-test:schema-3-test')
        self.assertEqual(KEY_FIELDS, ('transaction_id', 'model_version', 'feature_version'))

    def test_threshold_boundary_and_finite_range(self):
        self.assertEqual(build_response(transaction_id='T', risk_score=.7, threshold=.7,
                                        model_version='M', feature_version='F')['predicted_label'], 1)
        self.assertEqual(build_response(transaction_id='T', risk_score=0, threshold=.7,
                                        model_version='M', feature_version='F')['predicted_label'], 0)
        self.assertEqual(build_response(transaction_id='T', risk_score=1, threshold=1,
                                        model_version='M', feature_version='F')['predicted_label'], 1)

    def test_invalid_payloads_are_rejected(self):
        valid = {'transaction_id': 'TX001', 'risk_score': .9, 'predicted_label': 1,
                 'threshold': .7, 'model_version': 'M', 'feature_version': 'F'}
        invalid = [
            {k: v for k, v in valid.items() if k != 'feature_version'},
            dict(valid, extra=True),
            dict(valid, transaction_id=''),
            dict(valid, transaction_id='   '),
            dict(valid, risk_score=1.1),
            dict(valid, risk_score=-.1),
            dict(valid, risk_score=math.nan),
            dict(valid, threshold=math.inf),
            dict(valid, predicted_label=True),
            dict(valid, predicted_label=2),
            dict(valid, predicted_label=0),
            dict(valid, model_version=1),
            dict(valid, feature_version=None),
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_response(payload)
        with self.assertRaisesRegex(ValueError, 'object'):
            validate_response([])


if __name__ == '__main__':
    unittest.main()
