#!/usr/bin/env python3
"""Run dataset plots, frozen-GATv2 feature audit and criteria source-readiness EDA."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.config import load_config, project_path
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.evaluation.criteria_audit import run as criteria_audit
from fraud_pipeline.ingestion.request_loader import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/criteria-eda-v2.json')
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['eda_pipeline']
    output = project_path(settings['output'])
    if output.exists():
        parser.error('EDA output already exists; configure a new report version')
    # Validate the generated spec before creating outputs or invoking the expensive stages.
    subprocess.run([sys.executable, str(ROOT/'scripts/render_criteria_spec.py'), '--check'], check=True, cwd=ROOT)
    dataset = config['dataset']
    source, validation, features = (project_path(dataset[k]) for k in ('source', 'validation_report', 'features_output'))
    output.mkdir(parents=True)
    declaration = {'status': 'running', 'config_sha256': file_hash(project_path(args.config)),
                   'catalog_sha256': file_hash(project_path(settings['catalog'])),
                   'feature_manifest_sha256': file_hash(features/'manifest.json'),
                   'model_retraining': False, 'criterion_relabeling': False,
                   'test_policy': 'Coverage/plots are descriptive; feature associations use train, sensitivity uses validation; no new test scoring.'}
    (output/'run.json').write_text(json.dumps(declaration, indent=2)+'\n')
    try:
        print('Stage 1/3: audited dataset plots', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/plot_dataset.py'), '--data', str(source),
                        '--validation-report', str(validation), '--output', str(output/'dataset')], check=True, cwd=ROOT)
        print('Stage 2/3: feature EDA for the exact frozen GATv2 checkpoint', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_features.py'), '--config', str(project_path(args.config)),
                        '--output', str(output/'features')], check=True, cwd=ROOT)
        print('Stage 3/3: criteria source readiness', flush=True)
        criteria = criteria_audit(source, validation, TemporalDataset(features), project_path(settings['catalog']), output/'criteria')
        declaration.update(status='complete', readiness_counts=criteria['readiness_counts'])
        index = ['# Dataset EDA — criteria v2', '',
                 '[Dataset distribution and graph plots](dataset/dataset_eda.pdf)', '',
                 '[Feature leakage, correlation and validation sensitivity](features/report.md)', '',
                 '[Criteria source readiness](criteria/report.md)', '',
                 '[Criteria source/history coverage charts](criteria/criteria_readiness.pdf)', '',
                 'Sources present does not mean that a criterion is implemented or matched. No labels, model weights or thresholds were changed.']
        (output/'README.md').write_text('\n'.join(index)+'\n')
    except Exception as exc:
        declaration.update(status='failed', error=str(exc))
        raise
    finally:
        (output/'run.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'status': declaration['status'],
                      'criteria': declaration['readiness_counts']}, indent=2))


if __name__ == '__main__':
    main()
