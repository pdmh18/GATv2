import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fraud_pipeline.evaluation.dataset_baseline import CausalRules, binary_metrics, ranking_metrics


def sample(src='A', dst='B', amount=100, label=0):
    return dict(src_acc=src, dst_acc=dst, amount=amount, type='TRANSFER', timestamp='2025-01-01T10:00:00+07:00', channel='web', label=label)


class BaselineTests(unittest.TestCase):
    def test_scores_are_independent_of_ground_truth(self):
        first, second = CausalRules(), CausalRules()
        for i in range(50):
            self.assertEqual(first.score_then_observe(sample(label=0), i*60), second.score_then_observe(sample(label=1), i*60))

    def test_graph_score_uses_prior_unique_sources_and_current_edge_is_excluded(self):
        state = CausalRules()
        for i, src in enumerate(['A','B','C']):
            scores = state.score_then_observe(sample(src=src, dst='H'), i*20)
            self.assertEqual(scores['behavior_graph'], i/3)
        scores = state.score_then_observe(sample(src='D', dst='H'), 60)
        self.assertEqual(scores['behavior_graph'], 1)
        forwarding = state.score_then_observe(sample(src='H',dst='Z'), 80)
        self.assertEqual(forwarding['behavior_graph'], 4/3)
        old = state.score_then_observe(sample(src='X',dst='H'), 800)
        self.assertEqual(old['behavior_graph'], 0)

    def test_ranking_ties_and_confusion_matrix(self):
        rows = [(1,1,'s'), (1,0,'normal'), (0,0,'normal'), (0,1,'s')]
        self.assertEqual(ranking_metrics(rows), {'roc_auc': .5, 'average_precision': .5})
        m = binary_metrics(rows, .5)
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']), (1,1,1,1))
        self.assertEqual(m['f1'], .5)
