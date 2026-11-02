"""Publication exports from frozen evaluation predictions; no scoring or tuning."""
import json
from pathlib import Path
import numpy as np

LABELS = {'gatv2':'GATv2','gatv2_skip':'GATv2 Skip','feature_mlp':'Transaction MLP','endpoint_mlp':'Endpoint MLP',
          'gatv2_self_loops':'GAT self-loops','amount_only':'Amount / median','history_rule':'History rule'}


def curve_points(scores, labels):
    scores,labels=np.asarray(scores),np.asarray(labels)
    if scores.ndim!=1 or labels.shape!=scores.shape or not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid binary curve inputs')
    positives=int(labels.sum());negatives=len(labels)-positives
    if not positives or not negatives:
        raise ValueError('Curves require both classes')
    order=np.argsort(-scores,kind='stable');ordered=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(ordered)),len(scores)-1]
    tp=np.cumsum(labels[order])[ends];fp=ends+1-tp
    recall=tp/positives;precision=tp/(tp+fp)
    return {'fpr':np.r_[0,fp/negatives],'tpr':np.r_[0,recall],
            'recall':np.r_[0,recall],'precision':np.r_[1,precision],
            'ap':float(np.sum(np.diff(np.r_[0,recall])*precision)),
            'roc_auc':float(np.sum(np.diff(np.r_[0,fp/negatives])*(np.r_[0,recall][1:]+np.r_[0,recall][:-1])/2))}


def export_plots(run_directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..ingestion.request_loader import file_hash
    from .dataset_baseline import binary_metrics
    run_directory=Path(run_directory).resolve()
    report=json.loads((run_directory/'report.json').read_text())
    selection=json.loads((run_directory/'selection.json').read_text())
    if file_hash(run_directory/'selection.json')!=report['selection_sha256']:
        raise ValueError('Frozen selection checksum mismatch')
    selected=selection.get('reference_model','gatv2'); names=[selected];predictions={};curves={}
    if selected not in report['comparisons']:
        raise ValueError(f'Report requires {selected} test predictions as the research model')
    first=None
    for name in names:
        with np.load(run_directory/(name+'-test.npz'),allow_pickle=False) as saved:
            pred={k:saved[k].copy() for k in saved.files}
        if first is None:first=pred
        if not np.array_equal(pred['indices'],first['indices']) or not np.array_equal(pred['labels'],first['labels']):
            raise ValueError('Comparisons do not share test rows and labels')
        actual=binary_metrics([(float(s),int(y),'') for s,y in zip(pred['scores'],pred['labels'])],selection['models'][name]['threshold'])
        for field in ('tp','fp','fn','tn'):
            if actual[field]!=report['comparisons'][name]['test'][field]:raise ValueError('Prediction confusion differs from frozen report')
        curves[name]=curve_points(pred['scores'],pred['labels'])
        for field in ('ap','roc_auc'):
            if not np.isclose(curves[name][field],report['comparisons'][name]['test']['average_precision' if field=='ap' else field],atol=1e-10):
                raise ValueError('Curve metric disagrees with evaluation')
        predictions[name]=pred
    output=run_directory/'figures';output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':110})
    files=[];colors={name:plt.get_cmap('tab10')(i) for i,name in enumerate(names)}
    with PdfPages(output/'evaluation_charts.pdf') as combined:
        def save(fig,stem):
            fig.tight_layout()
            for extension in ('png','pdf'):
                path=output/(stem+'.'+extension)
                fig.savefig(path,dpi=220,bbox_inches='tight');files.append(path.name)
            combined.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for name in names:
            c=curves[name]
            axes[0].step(c['recall'],c['precision'],where='pre',color=colors[name],lw=1.8,
                         label=f'{LABELS.get(name,name)} (AP={c["ap"]:.4f})')
            axes[1].plot(c['fpr'],c['tpr'],color=colors[name],lw=1.8,
                         label=f'{LABELS.get(name,name)} (AUC={c["roc_auc"]:.4f})')
        prevalence=float(first['labels'].mean())
        axes[0].axhline(prevalence,color='black',ls='--',label=f'Prevalence = {prevalence:.2%}')
        axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision–Recall',xlim=(0,1),ylim=(0,1.02))
        axes[1].plot([0,1],[0,1],'k--',lw=1,label='Random ranking')
        axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC',xlim=(0,1),ylim=(0,1.02))
        for ax in axes:ax.legend(loc='lower left' if ax is axes[0] else 'lower right',fontsize=8)
        fig.suptitle(f'{LABELS.get(selected,selected)} · test ROC and Precision–Recall')
        save(fig,'roc_pr_curves')
        m=report['comparisons'][selected]['test']
        counts=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        proportions=counts/counts.sum(axis=1,keepdims=True)
        fig,ax=plt.subplots(figsize=(6.5,5.5))
        ax.imshow(proportions,cmap='Blues',vmin=0,vmax=1)
        for (i,j),count in np.ndenumerate(counts):
            ax.text(j,i,f'{count:,}\n({proportions[i,j]:.2%})',ha='center',va='center',fontsize=13,
                    color='white' if proportions[i,j]>.5 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],
               xlabel='Predicted label',ylabel='True label',title=f'{LABELS.get(selected,selected)} · test confusion')
        ax.grid(False)
        threshold=selection['models'][selected]['threshold']
        fig.text(.5,.01,f'Research model GATv2 · validation threshold {threshold:.4f} · count (% of true class)',ha='center',fontsize=9)
        save(fig,'confusion_matrix')
        fig,ax=plt.subplots(figsize=(10,2.5));ax.axis('off')
        columns=['Model','Precision','Recall','F1','AP','ROC-AUC','Threshold']
        rows=[]
        for name in names:
            m=report['comparisons'][name]['test']
            rows.append([LABELS.get(name,name)]+[f'{m[k]:.4f}' for k in ('precision','recall','f1','average_precision','roc_auc')]+
                        [f'{selection["models"][name]["threshold"]:.4f}'])
        table=ax.table(cellText=rows,colLabels=columns,cellLoc='center',loc='center',
                       colWidths=[.24,.125,.12,.12,.12,.125,.13])
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.7)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#d5dce3')
            if row==0:cell.set_facecolor('#e8edf2');cell.set_text_props(weight='bold')
            elif names[row-1]==selected:cell.set_facecolor('#e6f3f0');cell.set_text_props(weight='bold')
        ax.set_title(f'{LABELS.get(selected,selected)} · test metrics',pad=15)
        fig.text(.5,.035,'Threshold selected on validation and frozen before test.',
                 ha='center',fontsize=9)
        save(fig,'gatv2_metrics')
    manifest={'source_run':str(run_directory),'selection_sha256':report['selection_sha256'],
              'prediction_sha256':{name:file_hash(run_directory/(name+'-test.npz')) for name in names},
              'files':files+['evaluation_charts.pdf'], 'report_model':selected,
              'notes':['Saved test predictions only; no training, inference or threshold selection.',
                       'One confusion matrix for research model GATv2; counts and row percentages in the same cells.',
                       'PR and ROC share one figure; scalar metrics appear once in the GATv2 metrics table.',
                       'One synthetic seed; no uncertainty bounds or real-world generalization claim.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Evaluation report figures\n\n'+
                                   '\n'.join('- '+name for name in files if name.endswith('.png'))+
                                   '\n\nEach figure: PNG/PDF. Combined report: `evaluation_charts.pdf` (3 pages).\n'+
                                   'Confusion matrix: research model GATv2, counts and percentages by true class.\n')
    gat_report={'model':selected,'threshold_source':'Frozen validation F1 threshold',
                'checkpoint':selection['models'][selected], 'test':report['comparisons'][selected]['test'],
                'prediction_sha256':manifest['prediction_sha256'][selected],
                'limitations':'One synthetic dataset seed and one training seed; scores are not calibrated probabilities.'}
    (run_directory/'gatv2_report.json').write_text(json.dumps(gat_report,indent=2)+'\n')
    metrics=gat_report['test']
    (run_directory/'gatv2_report.md').write_text(f'# {LABELS.get(selected,selected)} test evaluation\n\n'+
        '| Precision | Recall | F1 | AP | ROC-AUC | Threshold |\n|---:|---:|---:|---:|---:|---:|\n'+
        '| '+' | '.join(f'{metrics[k]:.4f}' for k in ('precision','recall','f1','average_precision','roc_auc','threshold'))+' |\n\n'+
        f'TP={metrics["tp"]}, FP={metrics["fp"]}, FN={metrics["fn"]}, TN={metrics["tn"]}.\n\n'+
        'Checkpoint and threshold selected using validation before test. One synthetic seed; no real-world generalization claim.\n')
    return output
