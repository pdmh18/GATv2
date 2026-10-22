#!/usr/bin/env python3
"""Matched GATv2 device ablation; freeze both validation thresholds before test."""
import argparse
import copy
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.config import load_config,project_path
from fraud_pipeline.features.export import export_features
from fraud_pipeline.features.causal import names
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.ingestion.request_loader import file_hash
from fraud_pipeline.models.trainer import train_run
from fraud_pipeline.evaluation.phase5 import choose_threshold,metrics,run as evaluate_run
from fraud_pipeline.validation import validate_dataset
from fraud_pipeline.validation.report import markdown
from fraud_pipeline.split_export import export_splits


def run(config_path):
    config=load_config(config_path);d=config['dataset']
    source=project_path(d['source']);validation=project_path(d['validation_report'])
    splits=project_path(d['output']);features=project_path(d['features_output'])
    root=source.parent/'experiments/device-v1'
    if root.exists():raise ValueError('Device experiment exists; refusing adaptive rerun on the same test')
    root.mkdir(parents=True)
    print('Validating 500k source requests',flush=True)
    audit=validate_dataset(source)
    validation.parent.mkdir(parents=True,exist_ok=True)
    validation.write_text(json.dumps(audit,indent=2)+'\n');validation.with_suffix('.md').write_text(markdown(audit))
    if audit['error_count']:raise ValueError('Dataset audit failed')
    del audit;gc.collect()
    print('Exporting chronological split and expanded features',flush=True)
    export_splits(source,validation,splits,config['split'])
    manifest=export_features(source,validation,splits,features,config['features'])
    dataset=TemporalDataset(features)
    old_training=project_path('outputs/contextual-v4/seed-20261002/training/gatv2-v1')
    old=json.loads((old_training/'feature-manifest.json').read_text())
    mapping=[manifest['feature_names'].index(name) for name in old['feature_names']]
    # Compare the exact old input payload, not only amounts or label counts.
    with tempfile.TemporaryDirectory(prefix='legacy-input-check-',dir=root) as temporary:
        projected=Path(temporary)/'transaction_features.npy'
        np.save(projected,np.ascontiguousarray(dataset.arrays['transaction_features'][:,mapping],dtype='float32'))
        original_features_identical=file_hash(projected)==old['files']['transaction_features.npy']
    common=('labels.npy','split_codes.npy','request_timestamps_us.npy','target_edge_index.npy','graph_edge_index.npy',
            'graph_timestamps_us.npy','graph_prefix.npy','graph_kind.npy','node_features.npy','node_created_at_us.npy',
            'node_preprocessing.json','nodes.json')
    payload_matches={name:file_hash(features/name)==old['files'][name] for name in common}
    payload_matches['legacy_transaction_features_projection']=original_features_identical
    if not all(payload_matches.values()):raise ValueError('Core model inputs differ from historical baseline: '+str(payload_matches))
    device_names=[name for name in manifest['feature_names'] if name.startswith('platform_') or name in ('new_device','has_device','has_device_history','has_platform')]
    shared=source.parent/'training/shared-device-experiment/contexts'
    configs={}
    for key in ('without_device','with_device'):
        cfg=copy.deepcopy(config)
        cfg['training']['output']=str(root/key/'training')
        cfg['training']['context_cache']=str(shared)
        cfg['evaluation']['output']=str(root/key/'evaluation')
        cfg['training']['log_every_batches']=1000
        if key=='without_device':cfg['training']['model']['masked_feature_names']=device_names
        configs[key]=cfg
        (root/(key+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    protocol={'device_features':device_names,'feature_manifest_sha256':file_hash(features/'manifest.json'),
              'historical_input_checks':payload_matches,'selection_rule':'Validation AP; checkpoint per run selected by minimum validation weighted BCE; threshold maximum validation F1.',
              'budget':'Same dataset/split/seed 42/3 epochs/optimizer/architecture/parameter count; only device input mask differs.',
              'test_policy':'Both model checkpoints and thresholds frozen before any new test scoring. Historical baseline is descriptive; its test was previously opened.',
              'model_scope':'Only GATv2; no MLP baselines','status':'declared_before_training'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for key,cfg in configs.items():
        print('TRAIN',key,flush=True)
        train_run(features,Path(cfg['training']['output']),cfg['training'])
    choices={}
    for key,cfg in configs.items():
        path=Path(cfg['training']['output'])
        with np.load(path/'validation_predictions.npz') as saved:
            threshold=choose_threshold(saved['scores'],saved['labels'])
            choices[key]={'checkpoint_sha256':file_hash(path/'best.pt'),'threshold':threshold,
                          'validation':metrics(saved['scores'],saved['labels'],threshold)}
    selected=max(choices,key=lambda k:choices[k]['validation']['average_precision'])
    frozen={'choices':choices,'validation_selected':selected,'test_used_for_selection':False}
    (root/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n');frozen_digest=file_hash(root/'frozen.json')
    print('Both choices frozen before test:',selected,flush=True)
    comparisons={}
    for key,cfg in configs.items():
        path=Path(cfg['training']['output'])
        if file_hash(path/'best.pt')!=choices[key]['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
        result=evaluate_run(cfg,ROOT)
        observed=result['comparisons']['gatv2']
        if observed['validation']['threshold']!=choices[key]['threshold']:raise ValueError('Threshold changed after freeze')
        comparisons[key]=observed
    if file_hash(root/'frozen.json')!=frozen_digest:raise ValueError('Experiment selection changed')
    historical=json.loads(project_path('outputs/contextual-v4/seed-20261002/evaluation/phase5-v1/report.json').read_text())['comparisons']['gatv2']
    comparisons={'original_35_features':historical,**comparisons}
    full=dataset.arrays['transaction_features'];schema=manifest['feature_names'];coverage={}
    for split in ('train','validation','test'):
        mask=dataset.arrays[split+'_mask']
        coverage[split]={name:float(full[mask,schema.index(name)].mean()) for name in ('has_device','has_platform','new_device','has_authentication_verified')}
    report={'status':'complete','validation_selected':selected,'comparisons':comparisons,'device_coverage':coverage,
            'historical_inputs_identical':True,'frozen_sha256':frozen_digest,
            'limitations':['One dataset seed and one training seed; no generalization or statistical-significance claim.',
                           'Device context is synthetic, not real bank hardware fingerprinting.',
                           'Device metadata RNG does not read labels. Device is not itself an anomaly label.',
                           'New models use expanded features; direct device effect is with_device vs without_device, not vs original_35_features.',
                           'All reported test thresholds were selected on validation; test results do not change the frozen selection.']}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GATv2 device experiment','',f'Validation-selected run: {selected}. Same core historical inputs, verified by SHA256.','',
           '| GATv2 run | Validation AP | Test AP | Test precision | Test recall | Test F1 | FP | FN |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,row in comparisons.items():
        m=row['test'];lines.append(f'| {key} | {row["validation"]["average_precision"]:.4f} | {m["average_precision"]:.4f} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m["fp"]} | {m["fn"]} |')
    lines+=['','## Limits','']+['- '+s for s in report['limitations']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    plot(report,root)
    print(json.dumps(report,indent=2),flush=True)
    return report


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['comparisons'];labels=['Original 35','Expanded / device masked','Expanded / device enabled'];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric,title in zip(axes,('average_precision','f1'),('Average precision','F1 at validation threshold')):
        ax.bar(x-.18,[row['validation'][metric] for row in rows.values()],width=.36,label='Validation')
        ax.bar(x+.18,[row['test'][metric] for row in rows.values()],width=.36,label='Test')
        ax.set_xticks(x,labels,rotation=12,ha='right');ax.set(title=title,ylim=(0,1.04));ax.legend()
    fig.suptitle('GATv2 only · one seed · 500k requests · matched device ablation')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=200);fig.savefig(root/'comparison.pdf');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/device-v1.json')
    run(parser.parse_args().config)
