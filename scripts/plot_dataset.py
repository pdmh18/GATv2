"""Matplotlib EDA of the audited dataset; PNG figures and a multipage PDF."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.artifacts import defaults
from fraud_pipeline.ingestion.request_loader import jsonl_rows, verify_validation


def main():
    paths = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=paths['source'])
    parser.add_argument('--validation-report', type=Path, default=paths['validation'])
    parser.add_argument('--multiseed-report', type=Path, default=paths['multiseed'])
    parser.add_argument('--output', type=Path, default=paths['eda'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
                         'figure.dpi': 110, 'savefig.dpi': 180})
    manifest, hashes, audit = verify_validation(args.data, args.validation_report)
    labels = {r['transaction_id']: (r['is_anomaly'], r['scenario_id']) for _, r in jsonl_rows(args.data/'labels.jsonl')}
    daily = audit['summary']['daily']
    days = list(daily)
    type_names = ['TRANSFER', 'CASH_IN', 'CASH_OUT', 'PAYMENT', 'DEBIT']
    amounts, targets, kinds = [], [], []
    hours = np.zeros((2,24), dtype=int)
    account_counts, channel_counts, scenario_counts = Counter(), Counter(), Counter()
    pairs, receivers = Counter(), Counter()
    account_ids = {r['_id'] for _, r in jsonl_rows(args.data/'accounts.jsonl')}
    for _, tx in jsonl_rows(args.data/'transactions.jsonl'):
        target, scenario = labels[tx['transaction_id']]
        amounts.append(tx['amount']); targets.append(target); kinds.append(type_names.index(tx['type']))
        hours[int(target), int(tx['created_at'][11:13])] += 1
        account_counts[tx['source']['account_id']] += 1
        channel_counts[tx['channel']] += 1; scenario_counts[scenario] += 1
        src, dst = tx['source']['account_id'], tx['destination']['account_id']
        # Money-flow orientation for CASH_IN differs from raw source/destination roles.
        if tx['type'] == 'CASH_IN': src, dst = dst, src
        pairs[(src,dst)] += 1; receivers[dst] += 1
    amounts = np.asarray(amounts); targets = np.asarray(targets, dtype=bool); kinds = np.asarray(kinds)
    dates = np.asarray(days, dtype='datetime64[D]')
    volumes = np.asarray([daily[d]['count'] for d in days])
    anomalies = np.asarray([daily[d]['anomaly'] for d in days])
    failed = np.asarray([daily[d]['status'].get('failed',0) for d in days])
    out_neighbors, in_neighbors = defaultdict(set), defaultdict(set)
    for src,dst in pairs:
        out_neighbors[src].add(dst);in_neighbors[dst].add(src)
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    blue, orange, green = '#2563eb', '#ef4444', '#059669'
    def percent_ax(ax):
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    def title(fig, text):
        fig.suptitle(text, fontsize=17, fontweight='bold')
        fig.tight_layout(rect=(0,0.02,1,.95))
    def save(fig, name, pdf):
        path=args.output/(name+'.png');fig.savefig(path, bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight')
        files.append(path.name);plt.close(fig)
    with PdfPages(args.output/'dataset_eda.pdf') as pdf:
        fig,axes=plt.subplots(3,3,figsize=(17,13));ax=axes.ravel()
        ax[0].plot(dates,volumes,color=blue,lw=1);ax[0].set_title('Số giao dịch theo ngày');ax[0].tick_params(axis='x',rotation=35)
        rates=np.divide(anomalies*100,volumes,out=np.zeros_like(volumes,dtype=float),where=volumes>0)
        ax[1].plot(dates,rates,color=orange,alpha=.4,lw=.8,label='Theo ngày')
        roll_a=np.convolve(anomalies,np.ones(14),'valid');roll_n=np.convolve(volumes,np.ones(14),'valid')
        ax[1].plot(dates[13:],roll_a/roll_n*100,color=orange,lw=2,label='14 ngày (theo tổng count)')
        ax[1].axhline(2,color='gray',ls='--',label='Mục tiêu 2%');ax[1].set_title('Tỷ lệ anomaly');percent_ax(ax[1]);ax[1].legend(fontsize=8);ax[1].tick_params(axis='x',rotation=35)
        ax[2].plot(dates,failed/np.maximum(volumes,1)*100,color=orange);ax[2].set_title('Tỷ lệ thất bại (hậu kiểm)');percent_ax(ax[2]);ax[2].tick_params(axis='x',rotation=35)
        bins=np.linspace(np.log10(amounts.min()),np.log10(amounts.max()),55)
        for mask,name,color in [(~targets,'Normal',blue),(targets,'Anomaly',orange)]:
            ax[3].hist(np.log10(amounts[mask]),bins=bins,density=True,histtype='step',lw=1.8,color=color,label=name)
        ax[3].set_title('Phân bố amount theo nhãn');ax[3].set_xlabel('log10(amount VND)');ax[3].set_ylabel('Mật độ trong không gian log');ax[3].legend()
        counts=[int((kinds==i).sum()) for i in range(5)]
        ax[4].bar(type_names,counts,color=blue);ax[4].set_title('Số lượng theo nghiệp vụ');ax[4].tick_params(axis='x',rotation=25)
        type_rates=[float(targets[kinds==i].mean()*100) for i in range(5)]
        ax[5].bar(type_names,type_rates,color=orange);ax[5].set_title('Anomaly rate theo nghiệp vụ');percent_ax(ax[5]);ax[5].tick_params(axis='x',rotation=25)
        for i,name,color in [(0,'Normal',blue),(1,'Anomaly',orange)]:
            ax[6].plot(range(24),hours[i]/hours[i].sum()*100,color=color,label=name)
        ax[6].set_title('Giờ giao dịch: chuẩn hóa mỗi nhóm');ax[6].set_xlabel('Giờ (+07:00)');percent_ax(ax[6]);ax[6].legend()
        ax[7].bar(channel_counts.keys(),channel_counts.values(),color=green);ax[7].set_title('Phân bố channel')
        ax[8].hist(list(account_counts.values()),bins=35,color=blue);ax[8].set_title('Số request mỗi account');ax[8].set_xlabel('Giao dịch / account')
        title(fig,f'EDA — {len(amounts):,} giao dịch | {targets.sum():,} anomaly | seed {manifest["seed"]}')
        save(fig,'01_overview',pdf)

        fig,ax=plt.subplots(1,3,figsize=(18,5.5))
        for mask,name,color in [(~targets,'Normal',blue),(targets,'Anomaly',orange)]:
            sorted_amount=np.sort(amounts[mask]);ax[0].plot(sorted_amount,np.arange(1,len(sorted_amount)+1)/len(sorted_amount),label=name,color=color)
        ax[0].set_xscale('log');ax[0].set_xlabel('Amount VND (log scale)');ax[0].set_ylabel('CDF');ax[0].set_title('Phân phối tích lũy');ax[0].legend()
        ax[1].boxplot([amounts[kinds==i] for i in range(5)],tick_labels=type_names,showfliers=False)
        ax[1].set_yscale('log');ax[1].set_ylabel('Amount VND (log scale)');ax[1].set_title('Amount theo type (ẩn điểm ngoài whisker)');ax[1].tick_params(axis='x',rotation=25)
        names=[s for s in scenario_counts if s!='normal']
        ax[2].barh(names,[scenario_counts[s] for s in names],color=orange);ax[2].set_title('Nhãn scenario — sidecar');ax[2].set_xlabel('Số mẫu')
        title(fig,'Amount, nghiệp vụ và cơ cấu anomaly');save(fig,'02_amount_scenarios',pdf)

        fig,ax=plt.subplots(1,3,figsize=(17,5.5))
        heat=ax[0].imshow(hours/hours.sum(axis=1,keepdims=True)*100,aspect='auto',cmap='Blues')
        fig.colorbar(heat,ax=ax[0],label='Tỷ trọng (%)',shrink=.85)
        ax[0].set_yticks([0,1],['Normal','Anomaly']);ax[0].set_xticks(range(0,24,3));ax[0].set_xlabel('Giờ (+07:00)');ax[0].set_title('Tỷ trọng giờ trong từng nhóm (%)')
        x=np.sort(list(account_counts.values()));ax[1].plot(x,np.arange(1,len(x)+1)/len(x),color=blue)
        ax[1].set_title('CDF activity account');ax[1].set_xlabel('Số request');ax[1].set_ylabel('Tỷ lệ account')
        monthly=audit['summary']['monthly'];months=list(monthly)
        ax[2].plot(months,[monthly[m]['status'].get('failed',0)/monthly[m]['count']*100 for m in months],marker='o',color=orange)
        ax[2].set_title('Failure rate theo tháng');percent_ax(ax[2]);ax[2].tick_params(axis='x',rotation=35)
        title(fig,'Hành vi hoạt động và trạng thái tài chính');save(fig,'03_behavior_funding',pdf)

        fig,ax=plt.subplots(1,4,figsize=(21,5.5))
        degree_out=[len(out_neighbors[a]) for a in account_ids];degree_in=[len(in_neighbors[a]) for a in account_ids]
        ax[0].hist(degree_out,bins=30,color=blue,alpha=.6,label='Out-degree');ax[0].hist(degree_in,bins=30,color=orange,alpha=.5,label='In-degree')
        ax[0].set_title('Unique degree của account');ax[0].set_xlabel('Số đối tác khác nhau');ax[0].legend()
        repeated=np.asarray(list(pairs.values()));ax[1].hist(np.log10(repeated),bins=35,color=green)
        ax[1].set_yscale('log');ax[1].set_ylabel('Số cặp (thang log)')
        ax[1].set_title('Phân bố số request mỗi cặp có hướng');ax[1].set_xlabel('log10(số request): 0 = 1, 1 = 10, 2 = 100')
        singleton=int((repeated==1).sum())
        ax[1].text(.97,.97,f'{singleton:,}/{len(repeated):,} cặp chỉ có 1 request',
                   transform=ax[1].transAxes,ha='right',va='top',fontsize=8)
        top=sorted(receivers.items(),key=lambda item:item[1],reverse=True)[:15]
        ax[2].bar(range(len(top)),[n for _,n in top],color=blue);ax[2].set_title('15 node nhận nhiều nhất');ax[2].set_xlabel('Thứ hạng node (ẩn ID)')
        repeats=repeated[repeated>1]
        if len(repeats):
            ax[3].hist(np.log10(repeats),bins=25,color=green)
            ax[3].set_yscale('log')
        ax[3].set_title('Chỉ các cặp có ≥ 2 requests');ax[3].set_xlabel('log10(số request)');ax[3].set_ylabel('Số cặp (thang log)')
        title(fig,'Graph request: CASH_IN cash → account; các flow còn lại source → destination')
        save(fig,'04_graph_structure',pdf)

        runs=[]
        if args.multiseed_report.exists():
            candidates=json.loads(args.multiseed_report.read_text())
            # Keep experiments comparable to this dataset generation configuration.
            runs=[r for r in candidates if r.get('scenario_profile')==manifest.get('scenario_profile')] if any(r.get('dataset_sha256')==hashes for r in candidates) else []
        if runs:
            fig,ax=plt.subplots(1,2,figsize=(14,5.5));x=np.arange(len(runs));width=.25
            for i,(rule,label,color) in enumerate([('amount_only','Amount',blue),('behavior','Behavior',green),('behavior_graph','Behavior + graph',orange)]):
                values=[r['rules'][rule]['test']['f1'] for r in runs]
                ax[0].bar(x+(i-1)*width,values,width,label=label,color=color)
            ax[0].set_xticks(x,[str(r['seed']) for r in runs]);ax[0].set_ylim(0,1);ax[0].set_title('Diagnostic rule F1 trên test');ax[0].legend()
            for r in runs:
                m=r['monthly'];ax[1].plot(list(m),[m[k]['status'].get('failed',0)/m[k]['count']*100 for k in m],marker='o',label=str(r['seed']))
            ax[1].set_title('Failure rate theo tháng / seed');percent_ax(ax[1]);ax[1].tick_params(axis='x',rotation=35);ax[1].legend()
            title(fig,'Độ ổn định qua seed — rule tổng hợp, chưa phải kết quả GATv2')
            save(fig,'05_multiseed',pdf)
    summary={'dataset_sha256':hashes,'seed':manifest['seed'],'scenario_profile':manifest.get('scenario_profile'),
        'count':len(amounts),'anomaly_count':int(targets.sum()),'anomaly_rate':float(targets.mean()),
        'amount_quantiles':{name:dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(amounts[mask],[.5,.9,.95,.99,1])))) for mask,name in [(~targets,'normal'),(targets,'anomaly')]},
        'account_activity':{'min':min(account_counts.values()),'max':max(account_counts.values()),'median':float(np.median(list(account_counts.values())))},
        'hour_counts':{'normal':hours[0].tolist(),'anomaly':hours[1].tolist()},
        'anomaly_empty_hours':np.where(hours[1]==0)[0].tolist(),
        'max_activity_tie_count':sum(n==max(account_counts.values()) for n in account_counts.values()),
        'request_money_flow_graph':{'unique_directed_pairs':len(pairs),'account_count':len(account_ids),'cash_in_orientation':'cash → account'},
        'figures':files,'pdf':'dataset_eda.pdf',
        'notes':['Amounts/labels/status are post-hoc EDA; terminal status is not a training feature.',
                 'Histograms/hour curves normalize normal and anomaly separately to expose minority distribution.',
                 'Graph covers requests including failures, not only settled transfers.',
                 'Synthetic profiles and scenario assumptions are not empirical bank distributions.',
                 'Empty anomaly hours are a generator artifact and can create shortcut features; inspect before interpreting model performance.',
                 'Failure rates collapse after initial funding; not a calibrated real-bank failure distribution.']}
    (args.output/'eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
