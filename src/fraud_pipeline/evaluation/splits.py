"""Compare train/validation/test using one frozen GATv2 checkpoint and threshold."""
import json
from pathlib import Path
import numpy as np
from ..features.dataset import TemporalDataset
from ..ingestion.request_loader import file_hash
from ..models.batching import TemporalBatches
from ..registry.save_load import load
from .phase5 import predict, metrics


def evaluate_splits(config, root):
    import torch
    root=Path(root)
    run=root/config['evaluation']['output']
    output=run/'gatv2_partitions'
    if output.exists():
        raise ValueError('Partition evaluation exists; plotting can reuse it without scoring again')
    selection_path=run/'selection.json';digest=file_hash(selection_path)
    selection=json.loads(selection_path.read_text())
    identity=selection['models']['gatv2']
    checkpoint=Path(identity['checkpoint'])
    if file_hash(checkpoint)!=identity['sha256']:
        raise ValueError('Frozen GATv2 checkpoint changed')
    dataset=TemporalDataset(root/config['dataset']['features_output'])
    feature_digest=file_hash(dataset.directory/'manifest.json')
    if feature_digest!=selection['feature_manifest_sha256']:
        raise ValueError('Evaluation features differ from frozen selection')
    torch.set_num_threads(config['training']['threads'])
    model,metadata=load(checkpoint,expected_feature_sha256=feature_digest)
    batches=TemporalBatches(dataset,config['training']['snapshot_seconds'],config['training']['batch_size'],
                            Path(config['training'].get('context_cache',Path(identity['checkpoint']).parent/'contexts')))
    output.mkdir(parents=True)
    report={'model':'gatv2','checkpoint_sha256':identity['sha256'],'epoch':metadata['metadata']['epoch'],
            'threshold':identity['threshold'],'threshold_source':'Validation only, frozen before test',
            'selection_sha256':digest,'feature_manifest_sha256':feature_digest,'splits':{},
            'split_manifest':json.loads((root/config['dataset']['output']/'manifest.json').read_text())['split_rule'],
            'note':'Same final checkpoint, eval mode on all splits. Train metrics are in-sample; not generalization evidence.'}
    for split in ('train','validation','test'):
        print(f'Collecting GATv2 {split}',flush=True)
        if split=='train':
            pred,_=predict(model,batches,split)
        else:
            path=checkpoint.parent/'validation_predictions.npz' if split=='validation' else run/'gatv2-test.npz'
            with np.load(path,allow_pickle=False) as saved:
                pred={k:saved[k].copy() for k in saved.files}
        expected=np.flatnonzero(dataset.arrays[split+'_mask'])
        if not np.array_equal(pred['indices'],expected) or not np.array_equal(pred['labels'],dataset.arrays['labels'][expected]):
            raise ValueError('Split prediction coverage/labels mismatch')
        np.savez(output/(split+'.npz'),**pred)
        m=metrics(pred['scores'],pred['labels'],identity['threshold'])
        m['fraction_of_dataset']=len(expected)/dataset.manifest['request_count']
        m['anomaly_count']=int(pred['labels'].sum());m['anomaly_fraction']=float(pred['labels'].mean())
        report['splits'][split]=m
    if file_hash(selection_path)!=digest or file_hash(checkpoint)!=identity['sha256']:
        raise ValueError('Frozen selection/checkpoint changed during evaluation')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
