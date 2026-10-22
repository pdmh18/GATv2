import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from fraud_pipeline.evaluation.phase5 import choose_threshold
from fraud_pipeline.models.variants import build_model
from fraud_pipeline.evaluation.plots import curve_points


class EvaluationTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_with_ties(self):
        scores=np.array([.9,.7,.7,.5,.1]);labels=np.array([0,1,0,1,0])
        candidates=sorted(set(scores),reverse=True)
        def f1(t):
            pred=scores>=t
            return 2*sum(pred*labels)/(sum(pred)+sum(labels))
        expected=max(candidates,key=f1)
        self.assertEqual(choose_threshold(scores,labels),expected)
        self.assertRaises(ValueError,choose_threshold,[float('nan')],[1])
        self.assertRaises(ValueError,choose_threshold,[.5],[0])

    def test_curve_ties_and_average_precision(self):
        points=curve_points([.9,.5,.5,.1],[1,1,0,0])
        np.testing.assert_allclose(points['fpr'],[0,0,.5,1])
        np.testing.assert_allclose(points['tpr'],[0,.5,1,1])
        self.assertAlmostEqual(points['roc_auc'],.875)
        self.assertAlmostEqual(points['ap'],5/6)
        self.assertRaises(ValueError,curve_points,[.1,.2],[1,1])

    def test_device_mask_matches_zeroed_input_without_changing_parameters(self):
        import tempfile
        from fraud_pipeline.registry.save_load import save,load
        options=dict(kind='gatv2',in_dim=12,edge_dim=35,hidden_dim=32,heads=4,num_layers=2,dropout=0.)
        torch.manual_seed(42);original=build_model(options).eval()
        torch.manual_seed(42);masked=build_model(dict(options,masked_feature_indices=[2,4])).eval()
        self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in masked.parameters()))
        for name,value in original.state_dict().items():torch.testing.assert_close(value,masked.state_dict()[name])
        x=torch.randn(4,12);edges=torch.tensor([[0,1,2],[1,2,3]]);features=torch.randn(3,35);copy=features.clone()
        zeros=features.clone();zeros[:,[2,4]]=0
        expected=original(x=x,edge_index=edges,target_edge_index=edges,edge_features=zeros)
        actual=masked(x=x,edge_index=edges,target_edge_index=edges,edge_features=features)
        torch.testing.assert_close(actual,expected);torch.testing.assert_close(features,copy)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mask.pt'
            save(masked,path,model_config=dict(options,masked_feature_indices=[2,4]),metadata={'feature_manifest_sha256':'fixture'})
            restored,_=load(path,expected_feature_sha256='fixture')
            torch.testing.assert_close(restored(x=x,edge_index=edges,target_edge_index=edges,edge_features=features),expected)

    def test_ablation_does_not_depend_on_graph(self):
        torch.manual_seed(1)
        x=torch.randn(4,12);features=torch.randn(3,35)
        targets=torch.tensor([[0,1,2],[2,3,1]])
        edges=torch.tensor([[0,1,2],[1,2,3]])
        for kind in ('feature_mlp','endpoint_mlp','gatv2_self_loops'):
            model=build_model(dict(kind=kind,in_dim=12,edge_dim=35,hidden_dim=32,heads=4,num_layers=2,dropout=.2)).eval()
            with torch.no_grad():
                a=model(x=x,edge_index=edges,target_edge_index=targets,edge_features=features)
                b=model(x=x,edge_index=edges[:,:0],target_edge_index=targets,edge_features=features)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        model=build_model(dict(kind='feature_mlp',in_dim=12,edge_dim=35,hidden_dim=32,dropout=.2)).eval()
        torch.testing.assert_close(model(x=x,edge_index=edges,target_edge_index=targets,edge_features=features),model(x=x*100,edge_index=edges,target_edge_index=targets,edge_features=features))

if __name__=='__main__':
    unittest.main()
