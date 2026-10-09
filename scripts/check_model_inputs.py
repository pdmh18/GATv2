"""CPU forward/backward smoke check for the real GATv2 and causal tensors."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.config import load_config,project_path


def main():
    import numpy as np
    import torch
    import torch_geometric
    from fraud_pipeline.features.dataset import TemporalDataset
    from fraud_pipeline.models.gatv2 import GATv2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=project_path(load_config('configs/dataset.json')['dataset']['features_output']))
    args=parser.parse_args()
    dataset=TemporalDataset(args.features)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model=GATv2(in_dim=len(dataset.manifest['node_feature_names']),edge_dim=len(dataset.manifest['feature_names']))
    indices=[0,dataset.manifest['split_counts']['train']-1]
    losses=[]
    for index in indices:
        context=dataset.context(index)
        inputs={name:torch.from_numpy(context[name]).to(torch.float32 if name in ('x','edge_features') else torch.long)
                for name in ('x','edge_index','target_edge_index','edge_features')}
        logits=model(**inputs)
        if logits.shape!=(1,) or not torch.isfinite(logits).all():
            raise ValueError('Invalid GATv2 output')
        # Train rows only; this verifies gradients without optimizer updates or an evaluation claim.
        target=torch.tensor([float(dataset.arrays['labels'][index])])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Invalid GATv2 gradients')
        losses.append(float(loss.detach()))
    result={'status':'passed','torch':torch.__version__,'torch_geometric':torch_geometric.__version__,
            'in_dim':model.in_dim,'edge_dim':model.edge_dim,'train_rows_checked':indices,
            'forward_shape':[1],'finite_backward':True,'optimizer_updates':0,
            'note':'Compatibility smoke check only; no model training/evaluation result.'}
    output=args.features.parent/'validation/model-inputs.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
