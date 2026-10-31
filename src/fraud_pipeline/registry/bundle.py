"""Portable GATv2 serving bundle, with a validation-selected threshold."""
import json
import math
import shutil
import tempfile
from pathlib import Path

from ..ingestion.request_loader import file_hash
from .save_load import load

ARTIFACTS = ('manifest.json', 'preprocessing.json', 'node_preprocessing.json', 'nodes.json')
SERVING_MODEL_KINDS = ('gatv2', 'gatv2_skip')


def export_bundle(training, selection, output, *, model_version):
    training, selection, output = map(Path, (training, selection, output))
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError('model_version must be nonempty')
    if output.exists():
        raise ValueError('Bundle output exists; choose a new version')
    frozen = json.loads(selection.read_text())
    kind = frozen['selected_model']
    if frozen['status'] != 'frozen_before_test' or kind not in SERVING_MODEL_KINDS:
        raise ValueError(f'Serving requires a frozen GATv2 or GATv2 Skip selection: {kind}')
    if kind not in frozen['models']:
        raise ValueError(f'Frozen selection has no checkpoint for model kind: {kind}')
    chosen = frozen['models'][kind]
    threshold = chosen['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid validation threshold')
    if file_hash(training/'best.pt') != chosen['sha256']:
        raise ValueError('Checkpoint differs from frozen selection')
    digest = file_hash(training/'feature-manifest.json')
    if digest != frozen['feature_manifest_sha256']:
        raise ValueError('Selection belongs to a different feature artifact')
    model, checkpoint = load(training/'best.pt', expected_feature_sha256=digest)
    checkpoint_kind = checkpoint['model_config'].get('kind', 'gatv2')
    if checkpoint_kind != kind or model.use_skip_connection != (kind == 'gatv2_skip'):
        raise ValueError(f'Checkpoint architecture differs from frozen selection: {kind}')
    manifest = json.loads((training/'feature-manifest.json').read_text())
    if manifest['schema_version'] not in (2, 3):
        raise ValueError('Serving requires feature schema 2 or 3')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=output.parent) as temp:
        staged = Path(temp)/'bundle'
        staged.mkdir()
        for name in ARTIFACTS:
            source = training/('feature-'+name)
            if name != 'manifest.json' and file_hash(source) != manifest['files'][name]:
                raise ValueError(f'Training preprocessing checksum mismatch: {name}')
            shutil.copy2(source, staged/name)
        shutil.copy2(training/'best.pt', staged/'model.pt')
        shutil.copy2(selection, staged/'selection.json')
        files = {p.name: file_hash(p) for p in staged.iterdir()}
        metadata = {'schema_version': 1, 'model_kind': kind, 'model_version': model_version,
                    'feature_version': f"schema-{manifest['schema_version']}-{digest[:12]}",
                    'feature_manifest_sha256': digest, 'threshold': threshold,
                    'snapshot_seconds': checkpoint['metadata']['snapshot_seconds'],
                    'files': files,
                    'context_policy': 'First request in each epoch-aligned snapshot bucket; strict prior events only.',
                    'score_policy': 'Sigmoid of weighted-BCE logit; not a calibrated fraud probability.'}
        (staged/'bundle.json').write_text(json.dumps(metadata, indent=2)+'\n')
        staged.rename(output)
    return metadata
