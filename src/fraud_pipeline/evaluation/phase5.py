"""Validation-first comparisons; frozen choices precede all test scoring."""
import json
import time
from pathlib import Path

import numpy as np

from ..features.dataset import TemporalDataset
from ..ingestion.request_loader import file_hash, verify_validation
from ..models.batching import TemporalBatches
from ..models.trainer import train_run
from ..registry.save_load import load
from .dataset_baseline import binary_metrics, ranking_metrics


def choose_threshold(scores, labels):
    """Exact maximum validation F1; ties prefer the higher threshold."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all() or not labels.sum():
        raise ValueError('Threshold selection requires finite scores and positive binary labels')
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(scores)-1]
    tp = np.cumsum(labels[order])[ends]
    f1 = 2*tp/(ends+1+labels.sum())
    return float(ordered[ends[np.argmax(f1)]])


def metrics(scores, labels, threshold):
    rows = [(float(s), int(y), '') for s, y in zip(scores, labels)]
    result = binary_metrics(rows, threshold)
    result['count'] = len(rows)
    result['false_positive_rate'] = result['fp']/(result['fp']+result['tn']) if result['fp']+result['tn'] else None
    result.update(ranking_metrics(rows) if len(set(map(int, labels))) == 2 else
                  {'roc_auc': None, 'average_precision': None})
    return result


def predict(model, batches, split):
    import torch
    scores, indices, labels = [], [], []
    started = time.monotonic()
    forward_seconds = 0.
    model.eval()
    with torch.no_grad():
        for batch in batches.iterate(split):
            tensors = {k: torch.from_numpy(batch[k]).to(dtype=torch.float32 if k in ('x','edge_features') else torch.long)
                       for k in ('x','edge_index','edge_features','target_edge_index')}
            at = time.monotonic()
            output = torch.sigmoid(model(**tensors)).numpy()
            forward_seconds += time.monotonic()-at
            scores.extend(output); indices.extend(batch['indices']); labels.extend(batch['labels'])
    result = dict(scores=np.asarray(scores), indices=np.asarray(indices), labels=np.asarray(labels))
    expected = np.flatnonzero(batches.dataset.arrays[split+'_mask'])
    if not np.array_equal(result['indices'], expected) or not np.isfinite(result['scores']).all():
        raise ValueError('Prediction coverage/order or finite score failure')
    return result, dict(seconds=time.monotonic()-started, forward_seconds=forward_seconds,
                        forward_us_per_request=forward_seconds*1e6/len(indices),
                        note='Amortized CPU microbatch cost; excludes feature extraction. Wall time includes cache/context IO; not online request latency.')


def run(config, root):
    import torch
    root = Path(root).resolve()
    resolve = lambda p: root/p
    settings = config['evaluation']
    output = resolve(settings['output'])
    if output.exists():
        raise ValueError('Evaluation output already exists; use a new run path to avoid repeated test selection')
    features = resolve(config['dataset']['features_output'])
    _, source_hashes, _ = verify_validation(resolve(config['dataset']['source']), resolve(config['dataset']['validation_report']))
    reference = resolve(config['training']['output'])
    reference_config = json.loads((reference/'config.json').read_text())
    for key in ('seed','device','threads','epochs','batch_size','snapshot_seconds','learning_rate','weight_decay','gradient_clip','patience','min_delta','pos_weight','model'):
        if config['training'][key] != reference_config[key]:
            raise ValueError(f'Reference training budget differs: {key}')
    dataset = TemporalDataset(features)
    if source_hashes != dataset.manifest['source_sha256']:
        raise ValueError('Reporting raw source differs from feature snapshot')
    if str(resolve(config['dataset']['source']).resolve()) != dataset.manifest['source']:
        raise ValueError('Reporting source differs from feature artifact')
    digest = file_hash(features/'manifest.json')
    torch.set_num_threads(config['training']['threads'])
    output.mkdir(parents=True)
    cache = Path(config['training'].get('context_cache',reference/'contexts'))
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    variants = {'gatv2': reference}
    # Keep the Phase 4 checkpoint selection rule (minimum weighted validation loss).
    for kind in settings['train_variants']:
        run_path = output/kind
        options = dict(config['training'], model_kind=kind, context_cache=str(cache))
        print(f'Training comparison {kind}', flush=True)
        train_run(features, run_path, options)
        variants[kind] = run_path
    reports, predictions, identities = {}, {}, {}
    for kind, path in variants.items():
        model, checkpoint = load(path/'best.pt', expected_feature_sha256=digest)
        with np.load(path/'validation_predictions.npz') as saved:
            pred = {k: saved[k].copy() for k in saved.files}
        if not np.array_equal(pred['indices'], np.flatnonzero(dataset.arrays['validation_mask'])) or not np.array_equal(pred['labels'], dataset.arrays['labels'][pred['indices']]):
            raise ValueError('Validation predictions disagree with feature artifact')
        threshold = choose_threshold(pred['scores'], pred['labels'])
        reports[kind] = {'validation': metrics(pred['scores'], pred['labels'], threshold),
                         'parameter_count': sum(p.numel() for p in model.parameters()),
                         'best_epoch': checkpoint['metadata']['epoch'],
                         'training_seconds': json.loads((path/'summary.json').read_text())['seconds']}
        identities[kind] = {'checkpoint':str(path/'best.pt'), 'sha256':file_hash(path/'best.pt'), 'threshold':threshold}
        predictions[kind] = pred
    names = dataset.manifest['feature_names']
    rule_columns = [names.index(n) for n in settings['rule_features']]
    amount_column = names.index(settings['amount_feature'])
    def rule_scores(indices, kind):
        x = dataset.arrays['transaction_features'][indices]
        return x[:,amount_column] if kind == 'amount_only' else x[:,rule_columns].max(axis=1)
    val_indices = np.flatnonzero(dataset.arrays['validation_mask'])
    labels = dataset.arrays['labels'][val_indices]
    rule_kinds = ('amount_only','history_rule') if settings.get('include_rule_baselines', False) else ()
    for kind in rule_kinds:
        scores = rule_scores(val_indices, kind)
        threshold = choose_threshold(scores, labels)
        predictions[kind] = dict(indices=val_indices, scores=scores, labels=labels)
        reports[kind] = {'validation':metrics(scores,labels,threshold), 'parameter_count':0}
        identities[kind] = {'threshold':threshold, 'rule':'max of configured train-standardized causal columns' if kind=='history_rule' else settings['amount_feature']}
    # Main deployment candidate is selected only among trained models by validation F1.
    selected = max(variants, key=lambda k: (reports[k]['validation']['f1'], reports[k]['validation']['average_precision']))
    selection = {'status':'frozen_before_test', 'feature_manifest_sha256':digest,
                 'selection_rule':'Highest validation F1 after validation-only threshold optimization; AP breaks ties',
                 'selected_model':selected, 'models':identities, 'validation':reports,
                 'threshold_rule':'Exact maximum validation F1; ties choose higher threshold',
                 'seed':config['training']['seed'], 'test_used_for_selection':False,
                 'code_sha256':{str(p):file_hash(p) for p in [Path(__file__), Path(__file__).parents[1]/'models/variants.py']}}
    selection_path = output/'selection.json'
    selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    selection_digest = file_hash(selection_path)
    print(f'Frozen selection: {selected}. Beginning final test scoring.', flush=True)
    batches = TemporalBatches(dataset, config['training']['snapshot_seconds'], config['training']['batch_size'], cache, allow_test=True)
    for kind,path in variants.items():
        if file_hash(path/'best.pt') != identities[kind]['sha256']:
            raise ValueError('Frozen checkpoint changed')
        model,_ = load(path/'best.pt', expected_feature_sha256=digest)
        pred,timing = predict(model,batches,'test')
        np.savez(output/(kind+'-test.npz'), **pred)
        reports[kind]['test'] = metrics(pred['scores'],pred['labels'],identities[kind]['threshold'])
        reports[kind]['inference'] = timing
        print(f'{kind} test F1={reports[kind]["test"]["f1"]:.4f}', flush=True)
    test_indices = np.flatnonzero(dataset.arrays['test_mask'])
    for kind in rule_kinds:
        pred = dict(indices=test_indices, scores=rule_scores(test_indices,kind), labels=dataset.arrays['labels'][test_indices])
        np.savez(output/(kind+'-test.npz'), **pred)
        reports[kind]['test'] = metrics(pred['scores'],pred['labels'],identities[kind]['threshold'])
    with np.load(output/(selected+'-test.npz')) as saved:
        pred = {k:saved[k] for k in saved.files}
    # Scenario labels are joined only for reporting after model/threshold freezing.
    raw = resolve(config['dataset']['source'])
    scenarios = {r['event_id']:r['scenario_id'] for r in map(json.loads,(raw/'labels.jsonl').open())}
    trace = [json.loads(line)['event_id'] for line in (features/'requests.jsonl').open()]
    sample_rows = {r['event_id']:r for r in map(json.loads,(resolve(config['dataset']['output'])/'test.jsonl').open())}
    groups = {'type':{}, 'scenario':{}, 'month':{}}
    for position,index in enumerate(pred['indices']):
        event_id = trace[index]
        row = sample_rows[event_id]
        if row['label'] != int(pred['labels'][position]):
            raise ValueError('Reporting labels mismatch')
        for axis,value in [('type',row['type']),('scenario',scenarios[event_id]),('month',row['timestamp'][:7])]:
            groups[axis].setdefault(value,[]).append(position)
    breakdown = {axis:{value:metrics(pred['scores'][positions],pred['labels'][positions],identities[selected]['threshold'])
                      for value,positions in values.items()} for axis,values in groups.items()}
    if file_hash(selection_path) != selection_digest or file_hash(features/'manifest.json') != digest:
        raise ValueError('Frozen selection or feature identity changed')
    report = {'status':'complete', 'selected_model':selected,'selection_sha256':selection_digest,
              'comparisons':reports,'selected_test_breakdown':breakdown,
              'scenario_note':'Positive-only scenario cohorts: compare recall and FN; precision/F1 cannot measure false alarms in those cohorts.',
              'limitations':['One synthetic dataset seed and one training seed; no confidence intervals or multi-seed robustness claim.',
                             'Equal epochs/optimizer/split but architectures differ in capacity and compute.',
                             'Rule baseline uses train-standardized feature columns, not separately calibrated probabilities.',
                             'Historical neighborhood aggregates remain in transaction features of all models, including self-loop ablation.',
                             'Hourly graph cutoff sensitivity was not varied; context staleness remains a known limitation.',
                             'High synthetic metrics do not establish real banking generalization. Scores from weighted BCE are not calibrated probabilities.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    from .splits import evaluate_splits
    from .split_plots import export_split_plots
    evaluate_splits(config, root)
    export_split_plots(output)
    lines = ['# Phase 5 evaluation','',f'Selected on validation: **{selected}**. Thresholds frozen before test.','',
             '| Model | Validation F1 | Test AP | Test precision | Test recall | Test F1 |','|---|---:|---:|---:|---:|---:|']
    for kind,row in reports.items():
        t=row['test'];lines.append(f'| {kind} | {row["validation"]["f1"]:.4f} | {t["average_precision"]:.4f} | {t["precision"]:.4f} | {t["recall"]:.4f} | {t["f1"]:.4f} |')
    lines.extend(['','## Limits','']+['- '+s for s in report['limitations']])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report

