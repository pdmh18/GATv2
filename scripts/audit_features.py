#!/usr/bin/env python3
"""Train-only feature EDA; validation sensitivity; no model retraining."""
import argparse
import gc
import json
import re
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.config import load_config,project_path
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.evaluation.feature_audit import audit
from fraud_pipeline.ingestion.request_loader import file_hash


def rebuild(root,config):
    generator=ROOT.parent/'anomaly-data-pipeline'
    sys.path.insert(0,str(generator/'src'))
    from anomaly_data_pipeline.config import load_config as generator_config,load_pipeline_config
    from anomaly_data_pipeline.generation.pipeline import generate
    from fraud_pipeline.validation import validate_dataset
    from fraud_pipeline.split_export import export_splits
    from fraud_pipeline.features.export import export_features
    source=root/'raw';validation=root/'validation.json';splits=root/'splits';features=root/'features'
    base=generator_config(generator/'configs/base.yaml')
    print('Rebuilding temporary dataset',root,flush=True)
    generate(base.model_copy(update={'seed':config['dataset']['seed']}),source,load_pipeline_config(generator/'configs/pipeline.toml'));gc.collect()
    report=validate_dataset(source)
    validation.write_text(json.dumps(report,indent=2)+'\n')
    if report['error_count']:raise ValueError('Regenerated dataset failed validation')
    del report;gc.collect()
    export_splits(source,validation,splits,config['split']);gc.collect()
    export_features(source,validation,splits,features,config['features'])
    return features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/dataset.json')
    parser.add_argument('--features',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--overwrite',action='store_true',help='Regenerate audit reports in the same directory; input artifacts remain unchanged')
    parser.add_argument('--temporary-rebuild',action='store_true',help='Regenerate missing inputs in a temporary directory and clean them afterwards')
    args=parser.parse_args();config=load_config(args.config)
    settings=dict(config['feature_eda']);output=args.output or project_path(settings.pop('output'))
    settings.pop('output',None)
    evaluation=project_path(settings.pop('reference_evaluation_output',config['evaluation']['output']))
    training=project_path(settings.pop('reference_training_output',config['training']['output']))
    frozen=json.loads((evaluation/'selection.json').read_text())
    if file_hash(training/'best.pt')!=frozen['models']['gatv2']['sha256']:
        raise ValueError('Frozen GATv2 checkpoint changed')
    if file_hash(training/'feature-manifest.json')!=frozen['feature_manifest_sha256']:
        raise ValueError('Training schema differs from frozen selection')
    settings.update(snapshot_seconds=config['training']['snapshot_seconds'],threshold=frozen['models']['gatv2']['threshold'],
                    partition_report=str(evaluation/'gatv2_partitions/report.json'))
    for key in ('correlation_samples','node_contexts','importance_contexts','importance_repeats'):
        if type(settings[key]) is not int or settings[key]<1:parser.error(f'{key} must be a positive integer')
    if not 0<settings['correlation_threshold']<=1:parser.error('correlation_threshold must be in (0,1]')
    if output.exists():
        if not args.overwrite:parser.error('Audit output exists; use --overwrite or choose --output')
        if not (output/'report.json').is_file():parser.error('Refusing overwrite of an unrecognized audit directory')
    features=args.features or project_path(config['dataset']['features_output'])
    with ExitStack() as stack:
        if not features.exists():
            if not args.temporary_rebuild:parser.error('Feature tensors were cleaned; pass --temporary-rebuild or rebuild inputs first')
            root=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='gatv2-feature-audit-',dir='/tmp')))
            features=rebuild(root,config)
        output.mkdir(parents=True,exist_ok=True)
        print('Running causal regression checks',flush=True)
        checked=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_causal_features.py'],capture_output=True,text=True)
        log=checked.stdout+checked.stderr;(output/'causal_tests.txt').write_text(log)
        match=re.search(r'Ran (\d+) tests?',log)
        regression={'status':'passed' if checked.returncode==0 else 'failed','test_count':int(match.group(1)) if match else 0}
        if checked.returncode:raise ValueError('Causal regression checks failed; see causal_tests.txt')
        reproduction=json.loads((features.parent/'reproduction.json').read_text()) if (features.parent/'reproduction.json').exists() else None
        print('Computing feature EDA and GATv2 validation sensitivity',flush=True)
        report=audit(TemporalDataset(features),output,settings,training,reproduction=reproduction,regression=regression)
        # Keep a copy of the analyzed schema/provenance after temporary payload cleanup.
        (output/'feature-manifest.json').write_text((features/'manifest.json').read_text())
        (output/'audit_code.json').write_text(json.dumps({str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/'src/fraud_pipeline/evaluation/feature_audit.py']},indent=2)+'\n')
        print(json.dumps({'output':str(output),'leakage_checks':report['leakage']['status'],
                          'importance':report['importance']['status'],'ablation':'not_run'},indent=2),flush=True)

if __name__=='__main__':main()
