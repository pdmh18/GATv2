"""Compact train/validation/test report for the same GATv2 checkpoint."""
import json
from pathlib import Path
import numpy as np
from .plots import curve_points


def export_split_plots(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..ingestion.request_loader import file_hash
    from .dataset_baseline import binary_metrics
    run=Path(run);source=run/'gatv2_partitions'
    report=json.loads((source/'report.json').read_text())
    kind=report.get('model','gatv2')
    title='GATv2 Skip' if kind=='gatv2_skip' else 'GATv2'
    if file_hash(run/'selection.json')!=report['selection_sha256']:
        raise ValueError('Selection changed')
    output=run/'figures';output.mkdir(exist_ok=True)
    splits=('train','validation','test');colors=('steelblue','darkorange','seagreen');curves={}
    for split in splits:
        with np.load(source/(split+'.npz'),allow_pickle=False) as pred:
            c=curve_points(pred['scores'],pred['labels'])
            actual=binary_metrics([(float(s),int(y),'') for s,y in zip(pred['scores'],pred['labels'])],report['threshold'])
        m=report['splits'][split]
        for field in ('tn','tp','fp','fn'):
            if actual[field]!=m[field]:raise ValueError('Confusion mismatch')
        if not np.isclose(c['ap'],m['average_precision']) or not np.isclose(c['roc_auc'],m['roc_auc']):
            raise ValueError('Curve metric mismatch')
        curves[split]=c
    files=[]
    with PdfPages(output/'evaluation_charts.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout()
            for ext in ('png','pdf'):
                p=output/(name+'.'+ext);fig.savefig(p,dpi=220,bbox_inches='tight');files.append(p.name)
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for split,color in zip(splits,colors):
            c=curves[split]
            axes[0].step(c['recall'],c['precision'],where='pre',color=color,label=f'{split.title()} (AP={c["ap"]:.4f})')
            axes[1].plot(c['fpr'],c['tpr'],color=color,label=f'{split.title()} (AUC={c["roc_auc"]:.4f})')
        axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision–Recall',xlim=(0,1),ylim=(0,1.02))
        axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC',xlim=(0,1),ylim=(0,1.02))
        axes[1].plot([0,1],[0,1],'k--',lw=1,label='Random ranking')
        for ax in axes:ax.grid(alpha=.2);ax.legend(loc='lower left' if ax is axes[0] else 'lower right')
        fig.suptitle(f'{title} · same final checkpoint (epoch {report["epoch"]}) · train / validation / test')
        save(fig,'roc_pr_curves')
        fig,axes=plt.subplots(1,3,figsize=(13,4.6))
        for ax,split in zip(axes,splits):
            m=report['splits'][split];counts=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
            fractions=counts/counts.sum(axis=1,keepdims=True);ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
            for (i,j),count in np.ndenumerate(counts):
                ax.text(j,i,f'{count:,}\n({fractions[i,j]:.2%})',ha='center',va='center',color='white' if fractions[i,j]>.5 else 'black',fontsize=11)
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],
                   xlabel='Predicted label',ylabel='True label',title=f'{split.title()} · n={m["count"]:,}')
        fig.suptitle(f'{title} · confusion · fixed validation threshold {report["threshold"]:.4f}')
        fig.text(.5,.01,'Counts (% within true class). Train is in-sample; validation selects the threshold; test measures held-out performance.',ha='center',fontsize=9)
        save(fig,'confusion_matrix')
        fig,axes=plt.subplots(2,1,figsize=(12,5.2),gridspec_kw={'height_ratios':[1,3]})
        offset=0
        for split,color in zip(splits,colors):
            m=report['splits'][split];fraction=m['fraction_of_dataset']
            axes[0].barh([0],fraction,left=offset,color=color)
            axes[0].text(offset+fraction/2,0,f'{split.title()}\n{fraction:.2%} · {m["count"]:,}',ha='center',va='center',color='white');offset+=fraction
        axes[0].set_xlim(0,1);axes[0].axis('off')
        axes[0].set_title('Dataset partition · configured chronological ratio 50% / 30% / 20%')
        axes[1].axis('off')
        columns=['Split','Anomaly','Precision','Recall','F1','AP','ROC-AUC']
        rows=[]
        for split in splits:
            m=report['splits'][split]
            rows.append([split.title(),f'{m["anomaly_count"]:,} ({m["anomaly_fraction"]:.2%})']+[f'{m[k]:.4f}' for k in ('precision','recall','f1','average_precision','roc_auc')])
        table=axes[1].table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',colWidths=[.14,.22,.13,.13,.13,.13,.13]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#d5dce3')
            if row==0:cell.set_facecolor('#e8edf2');cell.set_text_props(weight='bold')
        fig.text(.5,.02,f'{title} · epoch {report["epoch"]} · threshold {report["threshold"]:.4f} on every split. Ratios by whole days; row counts can differ slightly.',ha='center',fontsize=9)
        save(fig,'gatv2_metrics')
    (output/'manifest.json').write_text(json.dumps({'model':kind,'partitions':list(splits),'files':files+['evaluation_charts.pdf'],
        'checkpoint_sha256':report['checkpoint_sha256'],'selection_sha256':report['selection_sha256'],
        'prediction_sha256':{split:file_hash(source/(split+'.npz')) for split in splits}},indent=2)+'\n')
    (output/'README.md').write_text('# GATv2 train / validation / test\n\nSame checkpoint and frozen validation threshold on all splits.\n\nROC/PR, confusion matrices and partition counts/metrics. PNG/PDF, combined 3-page PDF.\n')
    (run/'gatv2_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[f'# {title} train / validation / test','',
           f'Same best checkpoint: epoch {report["epoch"]}. Frozen validation threshold: {report["threshold"]:.4f}.','',
           '| Split | Requests | Actual fraction | Precision | Recall | F1 | AP | ROC-AUC |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in splits:
        m=report['splits'][split]
        lines.append('| '+split+' | '+f'{m["count"]:,} | {m["fraction_of_dataset"]:.2%} | '+
                     ' | '.join(f'{m[k]:.4f}' for k in ('precision','recall','f1','average_precision','roc_auc'))+' |')
    lines.extend(['','Chronological whole-day split: configured 50% / 30% / 20%; request counts can differ slightly.',
                  'Train metrics are in-sample. Validation selects checkpoint/threshold. Test is held out. One synthetic seed.'])
    (run/'gatv2_report.md').write_text('\n'.join(lines)+'\n')
    return output
