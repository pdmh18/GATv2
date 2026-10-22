"""Train-only feature EDA and validation permutation sensitivity for frozen GATv2.

Correlation and target association are screening signals, not leakage proofs.
Permutation importance is post-hoc sensitivity, not retrained ablation.
"""
import csv
import json
from pathlib import Path

import numpy as np
from .plots import curve_points
from .phase5 import metrics
from ..ingestion.request_loader import file_hash


def rank_columns(x):
    """Average ranks, including ties, without scipy."""
    result=np.empty(x.shape,dtype='float64')
    for col in range(x.shape[1]):
        order=np.argsort(x[:,col],kind='stable');values=x[order,col]
        starts=np.r_[0,np.flatnonzero(np.diff(values))+1]
        stops=np.r_[starts[1:],len(values)]
        result[order,col]=np.repeat((starts+stops-1)/2,stops-starts)
    return result


def correlations(x):
    """Constants have undefined correlations (NaN), including their diagonal."""
    def matrix(values):
        centered=values-values.mean(axis=0)
        norm=np.sqrt((centered**2).sum(axis=0))
        result=np.full((values.shape[1],values.shape[1]),np.nan)
        active=np.flatnonzero(norm>1e-12)
        normalized=centered[:,active]/norm[active]
        result[np.ix_(active,active)]=np.clip(normalized.T@normalized,-1,1)
        return result
    return matrix(x),matrix(rank_columns(x))


def feature_stats(x,names):
    rows=[]
    for i,name in enumerate(names):
        col=x[:,i];values,counts=np.unique(col,return_counts=True)
        rows.append({'feature':name,'count':len(col),'finite_fraction':float(np.isfinite(col).mean()),
                     'min':float(col.min()),'max':float(col.max()),'mean':float(col.mean()),
                     'std':float(col.std()),'unique':len(values),
                     'dominant_fraction':float(counts.max()/len(col)),
                     'constant':bool(len(values)==1),
                     'low_variation_candidate':bool(counts.max()/len(col)>=.995)})
    return rows


def redundant_pairs(x,names,pearson,spearman,threshold):
    rows=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            p,s=pearson[i,j],spearman[i,j]
            exact=bool(np.array_equal(x[:,i],x[:,j]))
            if exact or (np.isfinite(p) and abs(p)>=threshold) or (np.isfinite(s) and abs(s)>=threshold):
                left,lc=np.unique(x[:,i],return_counts=True);right,rc=np.unique(x[:,j],return_counts=True)
                modal=(x[:,i]==left[np.argmax(lc)]) & (x[:,j]==right[np.argmax(rc)])
                residual=x[~modal][:,[i,j]]
                rp=rs=None
                if len(residual)>2:
                    pm,sm=correlations(residual)
                    rp=float(pm[0,1]) if np.isfinite(pm[0,1]) else None
                    rs=float(sm[0,1]) if np.isfinite(sm[0,1]) else None
                rows.append({'a':names[i],'b':names[j],'pearson':float(p) if np.isfinite(p) else None,
                             'spearman':float(s) if np.isfinite(s) else None,'exact_on_sample':exact,
                             'joint_modal_fraction':float(modal.mean()),'non_joint_modal_rows':len(residual),
                             'non_joint_modal_pearson':rp,'non_joint_modal_spearman':rs})
    return rows


def feature_groups(names):
    groups={}
    for name in names:
        if name.startswith('authentication_') or name.startswith('has_authentication_'):group='authentication'
        elif name.startswith('platform_') or name in ('new_device','has_device','has_device_history','has_platform'):group='device_context'
        elif name in ('hour_sin','hour_cos','weekday_sin','weekday_cos','weekend'):group='calendar'
        elif name.startswith('type_') and name not in ('type_amount_reference_reliable','type_mean_reference_reliable'):group='transaction_type'
        elif name.startswith('channel_') and name!='channel_prior_fraction':group='channel_encoding'
        elif name in ('channel_prior_fraction','hour_prior_fraction'):group='behavior_habits'
        elif name=='new_receiver':group='receiver_novelty'
        elif name.startswith('log_src_request_count_'):group='request_velocity'
        elif name in ('log_src_history_count','log_seconds_since_previous','has_previous','cold_start'):group='source_history'
        elif name in ('log_dst_transfer_attempt_distinct_sources','log_src_transfer_attempt_distinct_sources',
                      'log_src_received_settled_transfer_amount','log_src_transfer_attempt_count'):group='transfer_neighborhood'
        else:group='amount_reference'
        groups.setdefault(group,[]).append(names.index(name))
    return groups


def write_csv(path,rows):
    if not rows:
        path.write_text('');return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def permutation_importance(dataset,training,settings,cache,original_manifest):
    """Permute feature groups jointly on validation, fixing graph/node embeddings."""
    import torch
    from ..registry.save_load import load
    from ..models.batching import TemporalBatches
    torch.set_num_threads(1)
    model,checkpoint=load(training/'best.pt',expected_feature_sha256=file_hash(original_manifest))
    if checkpoint['metadata']['snapshot_seconds']!=settings['snapshot_seconds']:
        raise ValueError('Snapshot policy differs from checkpoint')
    model.eval()
    batches=TemporalBatches(dataset,settings['snapshot_seconds'],128,cache)
    groups=list(batches.groups('validation'))
    positions=np.unique(np.linspace(0,len(groups)-1,min(settings['importance_contexts'],len(groups)),dtype=int))
    selected=[groups[i] for i in positions]
    indices=np.concatenate(selected);features=np.asarray(dataset.arrays['transaction_features'][indices]).copy()
    labels=np.asarray(dataset.arrays['labels'][indices]);contexts=[];offset=0
    with torch.no_grad():
        for group in selected:
            c=batches.context(group[0]);targets=dataset.arrays['target_edge_index'][:,group].copy()
            targets[~c['node_available_mask'][targets]]=0
            embedding=model.encode(torch.from_numpy(c['x']).float(),torch.from_numpy(c['edge_index']).long())
            contexts.append((offset,offset+len(group),embedding,torch.from_numpy(targets).long()));offset+=len(group)
        def scores(values):
            rows=[]
            for start,end,embedding,targets in contexts:
                rows.append(torch.sigmoid(model.classify_edges(embedding,targets,torch.from_numpy(values[start:end]).float())).numpy())
            return np.concatenate(rows)
        base=scores(features)
        with np.load(training/'validation_predictions.npz') as saved:
            if not np.array_equal(saved['indices'],np.flatnonzero(dataset.arrays['validation_mask'])):
                raise ValueError('Saved validation coverage differs')
            reference=saved['scores'][np.searchsorted(saved['indices'],indices)]
        if not np.allclose(base,reference,rtol=1e-5,atol=1e-6):
            raise ValueError('Validation subset does not reproduce frozen checkpoint predictions')
        baseline=metrics(base,labels,settings['threshold'])
        result=[]
        for name,columns in feature_groups(dataset.manifest['feature_names']).items():
            delta_ap=[];delta_f1=[]
            for repeat in range(settings['importance_repeats']):
                # Identical row permutation for every group in a repeat. One-hot groups stay valid.
                order=np.random.default_rng(settings['seed']+repeat).permutation(len(indices))
                changed=features.copy();changed[:,columns]=features[order][:,columns]
                m=metrics(scores(changed),labels,settings['threshold'])
                delta_ap.append(baseline['average_precision']-m['average_precision']);delta_f1.append(baseline['f1']-m['f1'])
            result.append({'group':name,'features':[dataset.manifest['feature_names'][i] for i in columns],
                           'ap_drop_mean':float(np.mean(delta_ap)),'ap_drop_std':float(np.std(delta_ap)),
                           'f1_drop_mean':float(np.mean(delta_f1)),'repeats':settings['importance_repeats']})
    return {'status':'complete','split':'validation','rows':len(indices),'positive_count':int(labels.sum()),
            'contexts':len(contexts),'sampling':'Uniformly spaced hourly buckets, all requests within each selected bucket',
            'baseline':baseline,'checkpoint_reproduction_verified':True,
            'groups':sorted(result,key=lambda r:r['ap_drop_mean'],reverse=True),
            'scope':'Transaction feature groups only; graph/node embeddings fixed. Not causal intervention or retrained ablation. Unconditional shuffles can break cross-group dependencies; AP drops are not additive or unique contributions.',
            'uncertainty':'Repeat std measures permutation variation, not a confidence interval or multi-seed robustness.'}


def audit(dataset,output,settings,training,*,reproduction=None,regression=None):
    from ..features.causal import names as expected_names
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    a=dataset.arrays;manifest=dataset.manifest
    train=np.flatnonzero(a['train_mask']);val=np.flatnonzero(a['validation_mask']);test=np.flatnonzero(a['test_mask'])
    names=manifest['feature_names'];node_names=manifest['node_feature_names']
    x=np.asarray(a['transaction_features'][train],dtype='float64')
    rng=np.random.default_rng(settings['seed']);sample=np.sort(rng.choice(len(x),min(settings['correlation_samples'],len(x)),replace=False))
    xs=x[sample];ys=np.asarray(a['labels'][train[sample]])
    pearson,spearman=correlations(xs)
    stats=feature_stats(x,names);pairs=redundant_pairs(xs,names,pearson,spearman,settings['correlation_threshold'])
    associations=[]
    for col,name in enumerate(names):
        if len(np.unique(xs[:,col]))<2:continue
        positive=curve_points(xs[:,col],ys);negative=curve_points(-xs[:,col],ys)
        best=max((positive,negative),key=lambda c:c['ap'])
        associations.append({'feature':name,'best_direction_train_ap':best['ap'],
                             'direction':'increasing' if best is positive else 'decreasing',
                             'train_auc_increasing':positive['roc_auc'],
                             'suspicious_association_candidate':bool(best['ap']>=.99)})
    # All observations are from prior histories of available nodes at train cutoffs.
    node_rows=[]
    selected=np.unique(np.linspace(0,len(train)-1,min(settings['node_contexts'],len(train)),dtype=int))
    for position in selected:
        c=dataset.context(int(train[position]));available=c['node_available_mask'].copy();available[0]=False
        node_rows.append(c['x'][available])
    nx=np.concatenate(node_rows).astype('float64')
    nstats=feature_stats(nx,node_names);np_,ns=correlations(nx)
    npairs=redundant_pairs(nx,node_names,np_,ns,settings['correlation_threshold'])
    masks=[np.asarray(a[s+'_mask']) for s in ('train','validation','test')]
    at=a['request_timestamps_us'];prefix=a['graph_prefix']
    prep=json.loads((dataset.directory/'preprocessing.json').read_text())
    nprep=json.loads((dataset.directory/'node_preprocessing.json').read_text())
    forbidden={'label','is_anomaly','scenario_id','risk','risk_score','posted_at','terminal_status','sourceBalanceAfter','destinationBalanceAfter'}
    checks={'feature_schema_matches_causal_builder':names==expected_names(manifest['options']),
            'no_forbidden_named_input':not forbidden.intersection(names+node_names),
            'finite_transaction_tensor':bool(np.isfinite(a['transaction_features']).all()),
            'split_masks_disjoint_and_complete':bool(np.all(sum(m.astype(int) for m in masks)==1)),
            'chronological_split':bool(at[train].max()<at[val].min() and at[val].max()<at[test].min()),
            'strictly_prior_graph_prefix':bool(np.array_equal(prefix,np.searchsorted(a['graph_timestamps_us'],at,side='left'))),
            'transaction_scaler_train_only_metadata':prep['fit_split']=='train' and prep['fit_count']==len(train),
            'node_scaler_train_only_metadata':nprep['fit_split']=='train' and all(a['train_mask'][i] for i in nprep['sampled_request_indices'])}
    numeric=prep['numeric_indices'];std=x[:,numeric].std(axis=0);active=std>1e-6
    checks['transaction_scaler_train_moments_match']=bool(np.allclose(x[:,numeric].mean(axis=0),0,atol=1e-4) and np.allclose(std[active],1,atol=1e-4))
    target=np.asarray(a['labels'][train]);direct=[]
    for col,name in enumerate(names):
        if np.array_equal(x[:,col],target) or np.array_equal(x[:,col],1-target):direct.append(name)
    checks['no_exact_label_copy_or_complement']=not direct
    report={'feature_windows':{k:v for k,v in manifest['options'].items() if 'window' in k},
            'request_context_coverage_train':{name:float(x[:,names.index(name)].mean()) for name in ('has_authentication_method','has_authentication_verified','has_device','has_platform') if name in names},
            'settings':settings,'feature_manifest_sha256':file_hash(dataset.directory/'manifest.json'),
            'selection_policy':'Feature screening/correlation uses train only. Permutation sensitivity uses validation only. Test metrics are descriptive, never used to select/drop features.',
            'leakage':{'checks':checks,'status':'checks_passed' if all(checks.values()) else 'failed_checks',
                       'causal_regression_tests':regression,'exact_target_copies':direct,
                       'conclusion':'These checks cannot prove absence of all leakage; live arrival-time availability and generator shortcuts remain separate risks.'},
            'transaction':{'count':len(x),'sample_count':len(xs),'stats':stats,'high_correlation_pairs':pairs,
                           'train_label_association':sorted(associations,key=lambda r:r['best_direction_train_ap'],reverse=True)},
            'node':{'context_count':len(selected),'observations':len(nx),'stats':nstats,'high_correlation_pairs':npairs,
                    'note':'Repeated available node observations at train cutoffs; not independent samples or endpoint-conditioned importance.'},
            'reproduction':reproduction,
            'ablation':{'status':'not_run','reason':'Requires retraining GATv2 per feature group with a frozen validation protocol. Permutation sensitivity is not ablation.'}}
    old_manifest=training/'feature-manifest.json'
    original=json.loads(old_manifest.read_text())
    payload_checks={name:file_hash(dataset.directory/name)==value for name,value in original['files'].items()}
    report['checkpoint_payload_checks']=payload_checks
    if all(payload_checks.values()) and names==original['feature_names'] and node_names==original['node_feature_names']:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='.contexts-',dir=output) as context_cache:
            report['importance']=permutation_importance(dataset,training,settings,context_cache,old_manifest)
    else:
        report['importance']={'status':'skipped','reason':'Feature payload differs from frozen checkpoint; refusing misleading importance.'}
    # Caller supplies the exact saved GATv2 partition report; no fresh test scoring.
    gap=Path(settings['partition_report'])
    if gap.exists():
        history=json.loads(gap.read_text())
        if history['feature_manifest_sha256']!=file_hash(old_manifest):
            raise ValueError('Saved partition report belongs to a different feature artifact')
        report['overfit']={'status':'descriptive_only' if all(payload_checks.values()) else 'historical_reference_only','checkpoint_sha256':history['checkpoint_sha256'],
                           'splits':history['splits'],'test_minus_train_f1':history['splits']['test']['f1']-history['splits']['train']['f1'],
                           'note':('A gap can reflect temporal distribution/history coverage, not only overfitting. Single seed and 3 epochs do not establish robustness.' if all(payload_checks.values()) else 'Historical scores belong to the old feature schema, not the expanded schema. Retrain GATv2 before judging the new features/windows.')}
        coverage=[]
        flags=[name for name in ('cold_start','has_previous','has_type_completed_history','type_amount_reference_reliable') if name in names]
        for split in ('train','validation','test'):
            mask=a[split+'_mask']
            row={'split':split,'count':int(mask.sum())}
            row.update({name:float(a['transaction_features'][mask,names.index(name)].mean()) for name in flags})
            coverage.append(row)
        report['overfit']['history_coverage']=coverage
        write_csv(output/'history_coverage.csv',coverage)
    else:report['overfit']={'status':'unavailable'}
    report['interpretation']=[
        'Constant unknown-category columns are reserved schema/fallback inputs; this dataset provides no training variation for them.',
        'High correlations may be driven by joint modal/zero values; inspect residual correlations before proposing removal.',
        'Permutation importance ranks dependence of this checkpoint on feature groups, not unique or causal feature value.',
        'Train/holdout history coverage differs; train below holdout alone does not establish absence of overfitting.',
        'Ablation and graph/node importance remain unmeasured; no feature has been removed.']
    write_csv(output/'transaction_stats.csv',stats);write_csv(output/'node_stats.csv',nstats)
    write_csv(output/'transaction_high_correlations.csv',pairs);write_csv(output/'node_high_correlations.csv',npairs)
    write_csv(output/'train_target_association.csv',report['transaction']['train_label_association'])
    np.savez(output/'correlations.npz',transaction_pearson=pearson,transaction_spearman=spearman,
             node_pearson=np_,node_spearman=ns,transaction_names=np.asarray(names),node_names=np.asarray(node_names))
    plot_report(report,spearman,ns,output)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Feature EDA — GATv2','',report['selection_policy'],'',
           'Window options (seconds): '+json.dumps(report['feature_windows']),
           'Optional request context coverage (train): '+json.dumps(report['request_context_coverage_train']), '',
           '## Leakage checks','']+['- '+name+': '+('PASS' if result else 'FAIL') for name,result in checks.items()]
    if regression:lines+=['',f'Causal fixture tests: {regression["status"]} ({regression["test_count"]} tests).']
    lines+=['','Passing checks does not prove absence of leakage. Review current-event availability and arrival order separately.',
            '','## Redundancy and low variation','',
            f'Transaction features: {len(names)}. Node features: {len(node_names)}.',
            f'Train transaction constants: {[r["feature"] for r in stats if r["constant"]]}.',
            f'Transaction high-correlation candidates: {len(pairs)}; node candidates: {len(npairs)}.',
            'Low-variation and correlated features are candidates for review, not automatic removals. One-hot/missing-history flags may remain meaningful. Pair CSV also reports correlation after excluding joint modal values to expose shared-zero/missingness effects.',
            '','## Validation permutation importance','',str(report['importance']['status'])]
    if report['importance']['status']=='complete':
        imp=report['importance'];lines+=['',f'Validation sample: {imp["rows"]} rows / {imp["contexts"]} hourly contexts; baseline AP {imp["baseline"]["average_precision"]:.4f}.',
                                       '', '| Group | Mean AP drop | Repeat std |','|---|---:|---:|']
        lines += [f'| {r["group"]} | {r["ap_drop_mean"]:.4f} | {r["ap_drop_std"]:.4f} |' for r in imp['groups']]
        lines+=['',imp['scope'],imp['uncertainty']]
    else:lines+=['',report['importance']['reason']]
    lines+=['','## Overfit screening','']
    if report['overfit']['status']=='descriptive_only':
        lines+=['| Split | Precision | Recall | F1 | AP |','|---|---:|---:|---:|---:|']
        for split,m in report['overfit']['splits'].items():
            lines+=['| '+split+' | '+' | '.join(f'{m[k]:.4f}' for k in ('precision','recall','f1','average_precision'))+' |']
        lines+=['',report['overfit']['note'],'','History coverage (fraction):','',
                '| Split | Cold start | Reliable same-type amount reference |','|---|---:|---:|']
        lines += [f'| {r["split"]} | {r.get("cold_start",0):.2%} | {r.get("type_amount_reference_reliable",0):.2%} |' for r in report['overfit']['history_coverage']]
    if report['overfit']['status']=='historical_reference_only':lines+=['',report['overfit']['note']]
    lines+=['','## Interpretation','']+['- '+s for s in report['interpretation']]
    lines+=['','## Not run','',report['ablation']['reason'],
            'No feature removal or tuning decision was made from test. No new models were trained.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def plot_report(report,tx_corr,node_corr,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output/'feature_audit.pdf') as combined:
        for kind,corr in [('transaction',tx_corr),('node',node_corr)]:
            names=[r['feature'] for r in report[kind]['stats']]
            labels=[n.replace('log_','').replace('src_','s_').replace('dst_','d_') for n in names]
            fig,ax=plt.subplots(figsize=(12,10) if kind=='transaction' else (8,7))
            ax.set_facecolor('#eeeeee');im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
            ax.set_xticks(range(len(names)),labels,rotation=90,fontsize=7)
            ax.set_yticks(range(len(names)),labels,fontsize=7)
            ax.set_title(f'{kind.title()} features · Spearman correlation (train only)\nGray = undefined for constant columns')
            fig.colorbar(im,ax=ax,fraction=.035);fig.tight_layout()
            fig.savefig(output/(kind+'_correlation.png'),dpi=180);combined.savefig(fig);plt.close(fig)
        imp=report['importance']
        if imp['status']=='complete':
            rows=imp['groups'][::-1];fig,ax=plt.subplots(figsize=(10,5.5))
            ax.barh([r['group'] for r in rows],[r['ap_drop_mean'] for r in rows],xerr=[r['ap_drop_std'] for r in rows],color='teal')
            ax.axvline(0,color='black',lw=.8);ax.set(xlabel='Validation AP drop after joint permutation (mean ± repeat std)',
                title=f'Frozen GATv2 · transaction-group sensitivity\n{imp["rows"]} validation rows · graph/node context fixed · not retrained ablation')
            fig.tight_layout();fig.savefig(output/'validation_importance.png',dpi=180);combined.savefig(fig);plt.close(fig)
