"""Dataset-grounded coverage audit of Phase 3 features and GATv2 inputs.

Window recommendations use train coverage, not held-out labels or model scores.
"""
import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict, deque
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.config import load_config, project_path
from fraud_pipeline.features.causal import COMPLETED, REQUESTS, TYPES, microseconds
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.ingestion.request_loader import jsonl_rows, verify_validation, file_hash


class History:
    def __init__(self, window):
        self.window = window * 1000000
        self.rows, self.values, self.total = deque(), [], 0
        self.cash_in = 0

    def add(self, at, amount, kind):
        self.rows.append((at, amount, kind))
        bisect.insort(self.values, amount)
        self.total += amount
        self.cash_in += kind == 'CASH_IN'

    def stats(self, at):
        while self.rows and self.rows[0][0] < at - self.window:
            _, value, kind = self.rows.popleft()
            self.values.pop(bisect.bisect_left(self.values, value))
            self.total -= value
            self.cash_in -= kind == 'CASH_IN'
        n = len(self.values)
        median = (self.values[(n-1)//2] + self.values[n//2])/2 if n else 0
        return n, self.total/n if n else 0, median, self.cash_in


class Graph:
    def __init__(self, window):
        self.window = window * 1000000
        self.rows, self.pairs, self.neighbors = deque(), Counter(), defaultdict(set)

    def add(self, at, src, dst):
        self.rows.append((at, src, dst))
        pair = tuple(sorted((src, dst)))
        self.pairs[pair] += 1
        self.neighbors[src].add(dst)
        self.neighbors[dst].add(src)

    def expire(self, at):
        while self.rows and self.rows[0][0] < at - self.window:
            _, src, dst = self.rows.popleft()
            pair = tuple(sorted((src, dst)))
            self.pairs[pair] -= 1
            if not self.pairs[pair]:
                del self.pairs[pair]
                self.neighbors[src].discard(dst)
                self.neighbors[dst].discard(src)


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dataset.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg, spec = config['dataset'], config['feature_audit']
    if config['features']['graph_history'] != 'settled' or not config['features']['bidirectional_graph']:
        raise ValueError('Graph audit supports settled, bidirectional model context only')
    source = project_path(cfg['source'])
    manifest, hashes, _ = verify_validation(source, project_path(cfg['validation_report']))
    dataset = TemporalDataset(project_path(cfg['features_output']))
    if hashes != dataset.manifest['source_sha256']:
        raise ValueError('Feature artifact and dataset disagree')
    prep = json.loads((dataset.directory/'preprocessing.json').read_text())
    output = args.output or dataset.directory.parent/'analysis/feature-suitability'
    output.mkdir(parents=True, exist_ok=True)
    windows, graph_windows, thresholds = spec['amount_windows_days'], spec['graph_windows_seconds'], spec['history_count_thresholds']
    size = len(dataset)
    codes = dataset.arrays['split_codes']
    labels = dataset.arrays['labels']
    histories = {w: defaultdict(lambda w=w: History(w*86400)) for w in windows}
    graphs = {w: Graph(w) for w in graph_windows}
    counts = np.zeros((size, len(windows), 2), dtype='int32')
    means, medians = np.zeros((size, len(windows), 2)), np.zeros((size, len(windows), 2))
    salary = np.zeros(size, dtype=bool)
    degree = np.zeros((size, len(graph_windows), 2), dtype='int16')
    homogeneous = np.zeros((size, 2), dtype=bool)
    kind_codes = np.zeros(size, dtype='uint8')
    node_types = {r['id']: r['type'] for r in json.loads((dataset.directory/'nodes.json').read_text())}
    type_amounts = defaultdict(list)
    per_account_counts = Counter()
    i = 0
    short_window = config['features'].get('neighborhood_window_seconds', config['features']['graph_window_seconds'])
    short = graphs[short_window]
    for at, group in groupby((event for _, event in jsonl_rows(source/'events.jsonl')),
                             key=lambda e: microseconds(e['occurred_at'])):
        events = [e for e in group if e['aggregate_type']=='transaction']
        if not events:
            continue
        for graph in graphs.values():
            graph.expire(at)
        for e in events:
            if e['event_type'] not in REQUESTS:
                continue
            p = e['payload']
            src, dst = p['sourceAccountId'], p['destinationAccountId']
            kind = 'TRANSFER' if e['event_type']=='TransferCreated' else p['transactionType']
            if at != dataset.arrays['request_timestamps_us'][i]:
                raise ValueError('Replay/feature timestamp alignment mismatch')
            kind_codes[i] = TYPES.index(kind)
            if codes[i] == 0:
                type_amounts[kind].append(p['amount'])
                per_account_counts[src] += 1
            for wi, window in enumerate(windows):
                for si, key in enumerate(((src, 'all'), (src, kind))):
                    n, mean, median, income = histories[window][key].stats(at)
                    counts[i,wi,si], means[i,wi,si], medians[i,wi,si] = n, mean, median
                    if window == 7 and si == 0:
                        salary[i] = income > 0
            for wi, window in enumerate(graph_windows):
                graph = graphs[window]
                degree[i,wi] = [len(graph.neighbors[src]), len(graph.neighbors[dst])]
            # The deployed encoder has two layers: inspect endpoint 2-hop type diversity.
            for endpoint, identity in enumerate((src,dst)):
                neighbors = short.neighbors[identity]
                region = {identity} | neighbors
                for neighbor in neighbors:
                    region.update(short.neighbors[neighbor])
                homogeneous[i,endpoint] = len({node_types[node] for node in region}) == 1
            i += 1
        for e in events:
            if e['event_type'] not in COMPLETED:
                continue
            p = e['payload']
            src, dst = p['sourceAccountId'], p['destinationAccountId']
            kind = 'TRANSFER' if e['event_type']=='TransferCompleted' else p['transactionType']
            for window in windows:
                histories[window][(src,'all')].add(at,p['amount'],kind)
                histories[window][(src,kind)].add(at,p['amount'],kind)
            if kind == 'CASH_IN':
                src,dst = dst,src
            for graph in graphs.values():
                graph.add(at,src,dst)
        if i and i % 100000 == 0:
            print(f'{i}/{size} requests audited',flush=True)
    if i != size:
        raise ValueError('Replay count mismatch')
    def quantiles(values):
        return dict(zip(('p10','p50','p90','p99'), map(float,np.quantile(values,[.1,.5,.9,.99])))) if len(values) else {}
    def coverage(mask):
        result = {}
        for wi, window in enumerate(windows):
            result[str(window)] = {}
            for si, scope in enumerate(('account_all_types','account_same_type')):
                values = counts[mask,wi,si]
                result[str(window)][scope] = {'count_quantiles':quantiles(values),
                    'fraction_at_least':{str(t):float((values>=t).mean()) for t in thresholds}}
        return result
    results = {'seed':manifest['seed'],'source_sha256':hashes,'analysis_config':spec,'count':size,
        'decision_policy':'Recommendations based on train support/architecture; held-out diagnostics do not select windows or model thresholds.',
        'splits':{},'train_amount_by_type':{},'feature_distributions':{},
        'model_contract':{'in_dim':len(dataset.manifest['node_feature_names']),
                          'edge_dim':len(dataset.manifest['feature_names']),
                          'encoder_layers':2,'edge_attributes_in_attention':False,
                          'node_features':dataset.manifest['node_policy'], 'schema_version':dataset.manifest['schema_version']}}
    wi7 = windows.index(7)
    for code, split in enumerate(('train','validation','test')):
        mask = codes == code
        typed = {}
        for k, kind in enumerate(TYPES[:-1]):
            typed[kind] = coverage(mask & (kind_codes==k))
        graph_stats = {}
        for wi, window in enumerate(graph_windows):
            values = degree[mask,wi]
            graph_stats[str(window)] = {'source_isolated_fraction':float((values[:,0]==0).mean()),
                'destination_isolated_fraction':float((values[:,1]==0).mean()),
                'both_isolated_fraction':float((values.max(axis=1)==0).mean()),
                'source_degree_quantiles':quantiles(values[:,0]),'destination_degree_quantiles':quantiles(values[:,1]),
                'by_type':{kind:{'source_isolated_fraction':float((degree[mask & (kind_codes==k),wi,0]==0).mean()),
                                'both_isolated_fraction':float((degree[mask & (kind_codes==k),wi].max(axis=1)==0).mean())}
                           for k,kind in enumerate(TYPES[:-1])}}
        eligible = mask & (counts[:,wi7,0] >= 5)
        ratio = means[eligible,wi7,0] / medians[eligible,wi7,0]
        same_eligible = mask & (counts[:,wi7,1] >= 5)
        same_ratio = means[same_eligible,wi7,1] / medians[same_eligible,wi7,1]
        paired_all_ratio = means[same_eligible,wi7,0] / medians[same_eligible,wi7,0]
        transfer = mask & (kind_codes==0)
        connected_transfer = transfer & (degree[:,graph_windows.index(short_window)].max(axis=1)>0)
        results['splits'][split] = {'count':int(mask.sum()),'coverage':coverage(mask),'coverage_by_type':typed,
            'mean_median_7d':{'eligible_min_history':5,'all_types_ratio_quantiles':quantiles(ratio),
                'all_types_fraction_mean_gt_2median':float((ratio>2).mean()),
                'same_type_ratio_quantiles':quantiles(same_ratio),
                'same_type_fraction_mean_gt_2median':float((same_ratio>2).mean()),
                'matched_eligible_count':int(same_eligible.sum()),
                'matched_all_types_fraction_mean_gt_2median':float((paired_all_ratio>2).mean()),
                'matched_same_type_fraction_mean_gt_2median':float((same_ratio>2).mean()),
                'all_history_has_cash_in_fraction':float(salary[mask].mean())},
            'graphs':graph_stats,
            'encoder_type_homogeneity':{'source_2hop_single_type_fraction':float(homogeneous[mask,0].mean()),
                'destination_2hop_single_type_fraction':float(homogeneous[mask,1].mean()),
                'connected_transfer_both_2hop_single_type_fraction':float(homogeneous[connected_transfer].all(axis=1).mean()) if connected_transfer.any() else None}}
    for kind, amounts in type_amounts.items():
        results['train_amount_by_type'][kind] = {'count':len(amounts),'quantiles':quantiles(amounts)}
    train = codes==0
    for index,name in enumerate(dataset.manifest['feature_names']):
        column = np.asarray(dataset.arrays['transaction_features'][:,index], dtype='float64')
        if index in prep['numeric_indices']:
            j = prep['numeric_indices'].index(index)
            column = column*prep['scale'][j]+prep['mean'][j]
        distribution = {}
        for code,split in enumerate(('train','validation','test')):
            values = column[codes==code]
            distribution[split] = {'quantiles':quantiles(values),'std':float(values.std()),
                'zero_fraction':float(np.isclose(values,0,atol=1e-6).mean()),'unique_values':int(len(np.unique(np.round(values,6))))}
        distribution['train_class_quantiles'] = {key:quantiles(column[train & (labels==target)]) for target,key in ((0,'normal'),(1,'anomaly'))}
        distribution['train_class_nonzero_fraction'] = {key:float((~np.isclose(column[train & (labels==target)],0,atol=1e-6)).mean()) for target,key in ((0,'normal'),(1,'anomaly'))}
        results['feature_distributions'][name] = distribution
    results['train_account_request_counts'] = quantiles(list(per_account_counts.values()))
    results['phase3_review'] = {
        'status': 'coverage_recheck_after_revision' if dataset.manifest['schema_version'] >= 2 else 'revision_required_before_training',
        'amount': 'Candidate: account+type median over 30 days, history count/reliability flags and current amount relative to median. Retain raw log amount; no oracle profiles or normal-only history.',
        'graph': 'Candidate: 24h settled message-passing graph; retain 600s attempt/fan-in/forwarding scalars as short-term head features. Connectivity is not a performance benchmark.',
        'nodes': 'Add causal incoming/outgoing activity, amount and degree features to type one-hot, evaluated at every context cutoff. Keep GATv2 encoder+MLP contract; derive in_dim/edge_dim from schema.',
        'constants': 'log_fee has no training signal in current dataset; consider excluding active input. Keep reserved unknown categories for contract handling.',
        'validation_needed': 'Re-export causal tensors after revision, rerun future/label invariance and Torch shape smoke test. Assess graph cost and tabular-vs-GAT ablation before claiming benefit.',
        'scope': 'This audit does not change exporter or model. Schema v2 implements the proposed input revision; performance and cost still require evaluation.' if dataset.manifest['schema_version'] >= 2 else 'Recommendations pending implementation.'}
    results['feature_manifest_sha256'] = file_hash(dataset.directory/'manifest.json')
    verify_validation(source, project_path(cfg['validation_report']))
    (output/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(output/'diagnostic_arrays.npz',counts=counts,means=means,medians=medians,
                        graph_degree=degree,split_codes=codes,type_codes=kind_codes,salary=salary)
    print(output/'analysis.json',flush=True)
    print(json.dumps(results['splits']['train']['coverage'],indent=2),flush=True)
    render(results,output)


def render(result,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    train = result['splits']['train']
    windows = result['analysis_config']['amount_windows_days']
    types = list(train['coverage_by_type'])
    with PdfPages(output/'feature_suitability.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(14,5.5))
        for wi,window in enumerate(windows):
            values=[train['coverage_by_type'][kind][str(window)]['account_same_type']['fraction_at_least']['5']*100 for kind in types]
            axes[0].bar(np.arange(len(types))+(wi-1)*.25,values,.25,label=f'{window} ngày')
        axes[0].set_xticks(np.arange(len(types)),types,rotation=25);axes[0].set_ylim(0,100);axes[0].legend();axes[0].set_ylabel('% requests train có ≥5 lịch sử cùng type')
        axes[0].set_title('Độ phủ tham chiếu amount theo account + type')
        keys=list(train['graphs'])
        axes[1].plot(keys,[train['graphs'][key]['source_isolated_fraction']*100 for key in keys],marker='o',label='Source cô lập')
        axes[1].plot(keys,[train['graphs'][key]['both_isolated_fraction']*100 for key in keys],marker='o',label='Cả hai đầu cô lập')
        axes[1].set_ylim(0,100);axes[1].set_xlabel('Window graph settled (giây)');axes[1].set_ylabel('% requests train');axes[1].legend();axes[1].set_title('Độ phủ graph tại thời điểm chấm')
        fig.suptitle('Đối chiếu window với dataset — quyết định dựa trên train',fontsize=16,fontweight='bold');fig.tight_layout()
        fig.savefig(output/'01_history_graph_coverage.png',dpi=200,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        constant=[name for name,stats in result['feature_distributions'].items() if stats['train']['std']<1e-7]
        fig,ax=plt.subplots(figsize=(14,7));ax.axis('off')
        lines=['GATv2: node type + causal history → 2 lớp attention → embeddings hai node + feature giao dịch → MLP' if result['model_contract']['schema_version'] >= 2 else 'GATv2: node type one-hot → 2 lớp attention → embeddings hai node + feature giao dịch → MLP',
            '',f"Source 2-hop chỉ có một loại node (train): {train['encoder_type_homogeneity']['source_2hop_single_type_fraction']:.1%}",
            f"Transfer có kết nối nhưng cả hai vùng 2-hop chỉ một loại: {train['encoder_type_homogeneity']['connected_transfer_both_2hop_single_type_fraction']:.1%}",
            '', 'Trong vùng node có vector đầu vào giống nhau, attention chuẩn hóa có thể tạo embeddings giống nhau.',
            'Vì vậy tăng window chưa đủ: cần node features có lịch sử causal khác nhau giữa các account.',
            '',f"Cùng nhóm ≥5 history cùng type, mean > 2×median khi gộp type: {train['mean_median_7d']['matched_all_types_fraction_mean_gt_2median']:.1%}",
            f"Trên nhóm đó, mean > 2×median khi tách đúng type: {train['mean_median_7d']['same_type_fraction_mean_gt_2median']:.1%}",
            '', 'Feature không đổi trên train: '+', '.join(constant),
            '', 'Đây là chẩn đoán dữ liệu/kiến trúc; chưa train hoặc đánh giá hiệu quả GATv2.']
        ax.text(.02,.94,'Kết quả đối chiếu với đầu vào model',fontsize=18,fontweight='bold',va='top')
        ax.text(.02,.84,'\n'.join(lines),fontsize=11,va='top',linespacing=1.7)
        fig.tight_layout();fig.savefig(output/'02_model_input_review.png',dpi=200,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    main()
