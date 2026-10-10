"""PyTorch GATv2 checkpoints with explicit architecture and artifact identity."""
from pathlib import Path


def save(model, path, *, model_config, metadata, optimizer=None):
    import torch
    path=Path(path)
    staged=path.with_suffix(path.suffix+'.tmp')
    payload={'model_config':model_config,'state_dict':model.state_dict(),'metadata':metadata}
    if optimizer is not None:
        payload['optimizer_state_dict']=optimizer.state_dict()
    torch.save(payload,staged)
    staged.replace(path)


def load(path, *, device='cpu', expected_feature_sha256=None):
    import torch
    from ..models.variants import build_model
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if expected_feature_sha256 is not None and checkpoint['metadata']['feature_manifest_sha256'] != expected_feature_sha256:
        raise ValueError('Checkpoint belongs to a different feature artifact')
    model=build_model(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    return model,checkpoint
