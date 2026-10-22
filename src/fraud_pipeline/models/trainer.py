"""Real GATv2 training on conservative causal snapshots; no test evaluation."""
import json
import random
import shutil
import time
from pathlib import Path
from uuid import uuid4

from ..features.dataset import TemporalDataset
from ..ingestion.request_loader import file_hash
from ..registry.save_load import load, save
from ..evaluation.dataset_baseline import binary_metrics, ranking_metrics
from .batching import TemporalBatches


def validate_config(config):
    for key in ('seed','threads','epochs','batch_size','snapshot_seconds','patience','log_every_batches'):
        if type(config[key]) is not int or config[key] < (0 if key=='seed' else 1):
            raise ValueError(f'Invalid training {key}')
    for key in ('learning_rate','gradient_clip'):
        if config[key]<=0:
            raise ValueError(f'{key} must be positive')
    if config['weight_decay']<0 or config['min_delta']<0 or not 0<=config['threshold']<=1:
        raise ValueError('Invalid training regularization/threshold')
    if config['pos_weight']!='train_ratio':
        raise ValueError('Only train_ratio pos_weight is supported')


def train_run(features, output, config, *, overwrite=False):
    import numpy as np
    import torch
    import torch_geometric
    from .variants import build_model
    validate_config(config)
    features,output=Path(features).resolve(),Path(output).resolve()
    if output==features or output in features.parents or features in output.parents:
        raise ValueError('Training output must be separate from feature inputs')
    dataset=TemporalDataset(features)
    digest=file_hash(features/'manifest.json')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    train_labels=np.asarray(dataset.arrays['labels'][dataset.arrays['train_mask']])
    positives=int(train_labels.sum());negatives=len(train_labels)-positives
    if not positives or not negatives:
        raise ValueError('Training requires both classes')
    weight=negatives/positives
    model_config=dict(config['model'],in_dim=len(dataset.manifest['node_feature_names']),edge_dim=len(dataset.manifest['feature_names']))
    masked_names=model_config.pop('masked_feature_names',[])
    if len(set(masked_names))!=len(masked_names) or any(name not in dataset.manifest['feature_names'] for name in masked_names):
        raise ValueError('Unknown/duplicate masked feature names')
    if masked_names:model_config['masked_feature_indices']=[dataset.manifest['feature_names'].index(name) for name in masked_names]
    model_config['kind']=config.get('model_kind','gatv2')
    model=build_model(model_config).to(device)
    if output.exists():
        if not overwrite:
            raise ValueError('Training run exists; choose a new run or --overwrite')
        if not (output/'config.json').exists():
            raise ValueError('Refusing overwrite of unrecognized run')
        output.rename(output.with_name(output.name+'.previous-'+uuid4().hex[:8]))
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ('manifest.json','preprocessing.json','node_preprocessing.json','nodes.json'):
        if (features/name).exists():
            shutil.copy2(features/name,output/('feature-'+name))
    batches=TemporalBatches(dataset,config['snapshot_seconds'],config['batch_size'],Path(config.get('context_cache',output/'contexts')))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    metadata={'feature_manifest_sha256':digest,'feature_directory':str(features),
              'model_config':model_config,'train_positives':positives,'train_negatives':negatives,'pos_weight':weight,
              'snapshot_seconds':config['snapshot_seconds'],
              'snapshot_policy':'Context at first request of each time bucket, reused conservatively. Exact transaction features. No later graph snapshot.',
              'torch':str(torch.__version__),'torch_geometric':str(torch_geometric.__version__),
              'code_sha256':{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('batching.py'),Path(__file__).with_name('gatv2.py'),Path(__file__).with_name('variants.py')]},
              'test_evaluated':False}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def pass_split(split, training, epoch):
        model.train(training)
        total,count=0.,0
        rows=[];indices=[]
        started=time.monotonic()
        for step,batch in enumerate(batches.iterate(split),1):
            tensors={name:torch.from_numpy(batch[name]).to(device=device,dtype=torch.float32 if name in ('x','edge_features') else torch.long)
                     for name in ('x','edge_index','target_edge_index','edge_features')}
            target=torch.from_numpy(batch['labels']).to(device=device,dtype=torch.float32)
            with torch.set_grad_enabled(training):
                logits=model(**tensors)
                loss=criterion(logits,target)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite model output/loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step()
            n=len(target);total+=float(loss.detach())*n;count+=n
            if not training:
                scores=torch.sigmoid(logits.detach()).cpu().numpy()
                rows.extend((float(score),int(label),'') for score,label in zip(scores,batch['labels']))
                indices.extend(map(int,batch['indices']))
            if step%config['log_every_batches']==0:
                print(f'epoch={epoch} {split} batches={step} samples={count} loss={total/count:.4f}',flush=True)
        if count!=dataset.manifest['split_counts'][split]:
            raise ValueError('Training/evaluation sample count mismatch')
        result={'loss':total/count,'count':count,'batches':step,'seconds':time.monotonic()-started}
        if not training:
            ranking=(ranking_metrics(rows) if len({row[1] for row in rows})==2
                     else {'roc_auc':None,'average_precision':None,'ranking_note':'Validation has only one class'})
            result['metrics']=dict(binary_metrics(rows,config['threshold']),**ranking)
        return result,rows,indices
    history=[];best=float('inf');bad=0;started=time.monotonic()
    try:
        for epoch in range(1,config['epochs']+1):
            train,_,_=pass_split('train',True,epoch)
            validation,rows,indices=pass_split('validation',False,epoch)
            history.append({'epoch':epoch,'train':train,'validation':validation})
            (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            save(model,output/'last.pt',model_config=model_config,metadata=dict(metadata,epoch=epoch),optimizer=optimizer)
            if validation['loss']<best-config['min_delta']:
                best=validation['loss'];bad=0
                save(model,output/'best.pt',model_config=model_config,metadata=dict(metadata,epoch=epoch),optimizer=optimizer)
                np.savez(output/'validation_predictions.npz',indices=np.asarray(indices),scores=np.asarray([r[0] for r in rows]),labels=np.asarray([r[1] for r in rows]))
            else:
                bad+=1
            ap=validation['metrics']['average_precision']
            print(f"epoch={epoch} complete train_loss={train['loss']:.4f} val_loss={validation['loss']:.4f} val_AP={ap}",flush=True)
            if bad>=config['patience']:
                break
        model,checkpoint=load(output/'best.pt',device=device,expected_feature_sha256=digest)
        validation,rows,indices=pass_split('validation',False,'reload')
        with np.load(output/'validation_predictions.npz') as saved:
            if not np.array_equal(saved['indices'],indices) or not np.allclose(saved['scores'],[r[0] for r in rows],atol=1e-7):
                raise ValueError('Reloaded checkpoint predictions disagree')
        if file_hash(features/'manifest.json')!=digest:
            raise ValueError('Feature manifest changed during training')
        for name,expected in dataset.manifest['files'].items():
            if file_hash(features/name)!=expected:
                raise ValueError('Feature input changed during training')
        summary={'status':'complete','best_epoch':checkpoint['metadata']['epoch'],'epochs_completed':len(history),
                 'best_validation':validation,'test_evaluated':False,'seconds':time.monotonic()-started,
                 'checkpoint_reload_verified':True,'model_config':model_config,
                 'limitations':'Synthetic dataset; hourly conservative graph snapshots. Validation threshold fixed at config value. No test evaluation or graph-vs-tabular ablation.'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        plot_history(history,output)
        print(json.dumps(summary,indent=2),flush=True)
        return summary
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'epochs_completed':len(history)},indent=2)+'\n')
        raise


def plot_history(history,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    epochs=[row['epoch'] for row in history]
    for split in ('train','validation'):
        axes[0].plot(epochs,[row[split]['loss'] for row in history],marker='o',label=split)
    axes[0].set(xlabel='Epoch',ylabel='Weighted BCE loss');axes[0].legend()
    axes[1].plot(epochs,[row['validation']['metrics']['average_precision'] for row in history],marker='o')
    axes[1].set(xlabel='Epoch',ylabel='Validation average precision',ylim=(0,1))
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=180);fig.savefig(output/'learning_curves.pdf');plt.close(fig)
