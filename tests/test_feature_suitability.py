import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_feature_suitability import Graph, History


class SuitabilityTests(unittest.TestCase):
    def test_history_duplicate_amounts_boundary_and_median(self):
        history = History(10)
        for at, amount, kind in [(0,10,'TRANSFER'),(1,10,'TRANSFER'),(2,1000,'CASH_IN')]:
            history.add(at*1000000,amount,kind)
        n,mean,median,cash = history.stats(10000000)
        self.assertEqual((n,median,cash),(3,10,1))
        self.assertEqual(mean,340)
        self.assertEqual(history.stats(11000000),(2,505,505,1))
        self.assertEqual(history.stats(13000000),(0,0,0,0))

    def test_bidirectional_graph_multiplicity_expiry(self):
        graph = Graph(10)
        graph.add(0,'A','B')
        graph.add(1000000,'B','A')
        graph.expire(10000000)
        self.assertEqual(graph.neighbors['A'],{'B'})
        graph.expire(11000000)
        self.assertEqual(graph.neighbors['A'],{'B'})
        graph.expire(12000000)
        self.assertEqual(graph.neighbors['A'],set())
        self.assertEqual(graph.neighbors['B'],set())
        self.assertEqual(len(graph.pairs),0)


if __name__=='__main__':
    unittest.main()
