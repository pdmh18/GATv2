import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fraud_pipeline.evaluation.feature_audit import rank_columns,correlations,feature_stats,redundant_pairs,feature_groups
from fraud_pipeline.features.causal import names
import json

class FeatureEDATests(unittest.TestCase):
    def test_ties_constants_and_duplicate_correlations(self):
        x=np.array([[1,2,7],[1,2,7],[3,6,7],[4,8,7]],dtype=float)
        np.testing.assert_allclose(rank_columns(x)[:,0],[.5,.5,2,3])
        p,s=correlations(x)
        self.assertAlmostEqual(p[0,1],1);self.assertAlmostEqual(s[0,1],1)
        self.assertTrue(np.isnan(p[2]).all());self.assertTrue(np.isnan(s[2]).all())
        stats=feature_stats(x,['a','b','constant']);self.assertTrue(stats[2]['constant'])
        pairs=redundant_pairs(x,['a','b','constant'],p,s,.95)
        self.assertEqual([(r['a'],r['b']) for r in pairs],[('a','b')])

    def test_shared_zero_correlation_does_not_imply_duplicate_information(self):
        x=np.vstack([np.zeros((1000,2)),[[1,9],[2,1],[3,8],[4,2]]])
        p,s=correlations(x)
        pair=redundant_pairs(x,['count','amount'],p,s,.95)[0]
        self.assertGreater(pair['spearman'],.99)
        self.assertGreater(pair['joint_modal_fraction'],.99)
        self.assertFalse(pair['exact_on_sample'])
        self.assertLess(pair['non_joint_modal_spearman'],0)

    def test_grouping_covers_schema_without_breaking_categories(self):
        root=Path(__file__).resolve().parents[1]
        schema=names(json.loads((root/'configs/dataset.json').read_text())['features'])
        groups=feature_groups(schema)
        self.assertEqual(sorted(i for cols in groups.values() for i in cols),list(range(len(schema))))
        self.assertTrue(all(schema[i].startswith('type_') for i in groups['transaction_type']))
        self.assertNotIn(schema.index('type_amount_reference_reliable'),groups['transaction_type'])
        self.assertEqual(len(groups['transaction_type']),6);self.assertEqual(len(groups['channel_encoding']),4)

    def test_expanded_mean_reliability_is_not_a_type_category(self):
        root=Path(__file__).resolve().parents[1]
        schema=names(json.loads((root/'configs/review-v2.json').read_text())['features'])
        groups=feature_groups(schema)
        self.assertEqual(len(groups['transaction_type']),6)
        self.assertIn(schema.index('type_mean_reference_reliable'),groups['amount_reference'])
        self.assertNotIn(schema.index('type_mean_reference_reliable'),groups['transaction_type'])

if __name__=='__main__':unittest.main()
