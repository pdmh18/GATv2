"""External holdout evaluation for AnomalyBank GATv2 Skip v2 (5 epochs).

Run inside the existing Google Colab runtime after generating 200k raw, splits,
features and features-frozen-500k. No training or threshold fitting is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/content/GATv2')
EXT = REPO / 'outputs/external-test-200k/seed-20261120'
OLD = REPO / 'outputs/contextual-v5-device/seed-20261002/features-v3/skip-v2'
FROZEN = EXT / 'features-frozen-500k'
RUN = Path('/content/drive/MyDrive/AnomalyBank/GATv2/skip-v2-runs/skip-v2-run-01')
DEST = RUN / 'external-test-200k-seed-20261120'
KIND = 'gatv2_skip'
EXPECTED_REQUESTS = 200_000
EXPECTED_POSITIVES = 4_000


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def check(cond, why):
    if not cond:
        raise RuntimeError(why)


def main():
    import torch
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, f1_score,
        precision_score, precision_recall_curve, recall_score,
        roc_auc_score, roc_curve,
    )
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / 'src'))
    from fraud_pipeline.features.dataset import TemporalDataset
    from fraud_pipeline.models.batching import TemporalBatches
    from fraud_pipeline.registry.save_load import load

    # Check the 500k feature artifact and the frozen validation-only selection.
    check(FROZEN.is_dir() and OLD.is_dir(), 'Missing feature artifacts. Run earlier Colab cells first.')
    check(not DEST.exists(), f'External evaluation already exists; refusing overwrite: {DEST}')
    selection = json_read(RUN / 'evaluation/selection.json')
    check(selection['status'] == 'frozen_before_test', 'Selection is not frozen before test')
    check(selection['selected_model'] == KIND, 'Selected architecture differs from GATv2 Skip')
    selected = selection['models'][KIND]
    threshold = float(selected['threshold'])
    check(np.isfinite(threshold) and 0 <= threshold <= 1, 'Invalid original threshold')
    old_digest = sha256(OLD / 'manifest.json')
    check(old_digest == selection['feature_manifest_sha256'], 'Original feature manifest differs from frozen selection')
    check(old_digest == sha256(RUN / 'training/feature-manifest.json'), 'Training feature manifest disagrees with original')
    best = RUN / 'training/best.pt'
    check(sha256(best) == selected['sha256'], 'Checkpoint SHA-256 does not match frozen selection')
    train_cfg = json_read(RUN / 'training/config.json')
    check(train_cfg.get('model_kind') == KIND and int(train_cfg['epochs']) == 5, 'Not the requested Skip v2 five-epoch run')

    old_manifest = json_read(OLD / 'manifest.json')
    manifest_file = FROZEN / 'manifest.json'
    manifest = json_read(manifest_file)
    for field in ('feature_names', 'node_feature_names', 'schema_version', 'options'):
        check(manifest[field] == old_manifest[field], f'Old/new features incompatible: {field}')
    check(manifest['request_count'] == EXPECTED_REQUESTS, 'Unexpected number of new transactions')
    old_p = json_read(OLD / 'preprocessing.json')
    new_p = json_read(FROZEN / 'preprocessing.json')
    old_np = json_read(OLD / 'node_preprocessing.json')
    new_np = json_read(FROZEN / 'node_preprocessing.json')
    check(old_p == new_p and old_np == new_np, 'Frozen feature files do not contain ORIGINAL preprocessing parameters')
    check(len(old_p['numeric_indices']) == 17 and len(old_np['numeric_indices']) == 6, 'Unexpected scaler dimensions')

    # Previous conversion already transformed transaction_features.npy; DO NOT repeat.
    # Its old manifest hashes must now be refreshed to describe these NEW bytes.
    changed = ('transaction_features.npy', 'preprocessing.json', 'node_preprocessing.json')
    for filename in changed:
        check(filename in manifest['files'], f'Feature manifest does not track {filename}')
        manifest['files'][filename] = sha256(FROZEN / filename)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # TemporalDataset verifies EVERY file digest in the external feature artifact.
    dataset = TemporalDataset(FROZEN)
    check(len(dataset) == EXPECTED_REQUESTS, 'Dataset loader count mismatch')
    labels = np.asarray(dataset.arrays['labels'], dtype=np.int8)
    check(len(labels) == EXPECTED_REQUESTS and int(labels.sum()) == EXPECTED_POSITIVES, 'External labels incorrect')
    check(np.isin(labels, (0, 1)).all(), 'Nonbinary labels')
    check(np.isfinite(dataset.arrays['transaction_features']).all(), 'Nonfinite transaction features')
    masks = [np.asarray(dataset.arrays[s + '_mask'], dtype=bool) for s in ('train', 'validation', 'test')]
    check(np.all(np.sum(masks, axis=0) == 1), 'Split masks do not cover each request exactly once')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(int(train_cfg.get('threads', 1)))
    model, ckpt = load(best, device=device, expected_feature_sha256=old_digest)
    check(ckpt['model_config'].get('kind') == KIND and model.use_skip_connection, 'Incorrect checkpoint architecture')
    model.eval()
    print(f'MODEL={KIND}; DEVICE={device}; THRESHOLD={threshold:.14f}', flush=True)
    print(f'EXTERNAL DATASET={len(dataset):,}; ANOMALIES={int(labels.sum()):,}', flush=True)
    print('Running inference: train/validation/test masks are only coverage partitions, NOT training steps.', flush=True)

    # This is a NEW cache tied to the new feature manifest. Never use original 500k cache.
    batches = TemporalBatches(dataset, int(train_cfg['snapshot_seconds']),
                              int(train_cfg['batch_size']), EXT / 'context-cache-frozen-500k',
                              allow_test=True)
    scores = np.full(len(dataset), np.nan, dtype=np.float32)
    visited = np.zeros(len(dataset), dtype=bool)
    started = time.monotonic()
    with torch.inference_mode():
        for split in ('train', 'validation', 'test'):
            seen_in_split = 0
            for batch in batches.iterate(split):
                tensors = {k: torch.as_tensor(batch[k], device=device,
                                              dtype=torch.float32 if k in ('x', 'edge_features') else torch.long)
                           for k in ('x', 'edge_index', 'edge_features', 'target_edge_index')}
                p = torch.sigmoid(model(**tensors)).detach().cpu().numpy().reshape(-1)
                indices = np.asarray(batch['indices'], dtype=np.int64)
                check(len(p) == len(indices) and np.isfinite(p).all(), 'Bad prediction output')
                check(not visited[indices].any(), 'Duplicate request prediction')
                scores[indices] = p
                visited[indices] = True
                seen_in_split += len(indices)
                if seen_in_split % 10000 < len(indices):
                    print(f'{split}: {seen_in_split:,} scored', flush=True)
            check(seen_in_split == int(masks[('train', 'validation', 'test').index(split)].sum()), f'{split} coverage failure')
            print(f'{split} DONE: {seen_in_split:,}', flush=True)

    check(visited.all() and np.isfinite(scores).all(), 'Not all 200,000 requests received predictions')
    pred = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, pred, labels=[0, 1]).ravel())
    report = {
        'model': KIND,
        'count': int(len(labels)), 'positives': int(labels.sum()),
        'threshold': threshold,
        'threshold_source': 'Frozen validation-only threshold from the 500k original model run',
        'checkpoint_sha256': sha256(best),
        'original_feature_manifest_sha256': old_digest,
        'external_feature_manifest_sha256': sha256(manifest_file),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': float(precision_score(labels, pred, zero_division=0)),
        'recall': float(recall_score(labels, pred, zero_division=0)),
        'f1': float(f1_score(labels, pred, zero_division=0)),
        'average_precision': float(average_precision_score(labels, scores)),
        'roc_auc': float(roc_auc_score(labels, scores)),
        'seconds': round(time.monotonic() - started, 2),
        'note': 'Synthetic external-seed holdout from same data generator; no training, retuning, or scaler fitting on the external set. The original pipeline partitions are used only to enumerate all requests.'
    }

    DEST.mkdir(parents=True, exist_ok=False)
    (DEST / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    np.savez_compressed(DEST / 'predictions.npz', indices=np.arange(len(dataset)), scores=scores,
                        labels=labels, predicted=pred)
    with (FROZEN / 'requests.jsonl').open(encoding='utf-8') as fin, (DEST / 'predictions.csv').open('w', encoding='utf-8', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(('row', 'transaction_id', 'event_id', 'label', 'risk_score', 'prediction'))
        count = 0
        for i, line in enumerate(fin):
            row = json.loads(line)
            check(int(row['row']) == i, 'Request ID order mismatch')
            w.writerow((i, row['transaction_id'], row['event_id'], int(labels[i]), f'{scores[i]:.8f}', int(pred[i])))
            count += 1
        check(count == len(labels), 'Request metadata does not cover 200k records')

    fpr, tpr, _ = roc_curve(labels, scores)
    rec, prec, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr); ax[0].set(xlabel='False positive rate', ylabel='True positive rate',
                                      title=f'External ROC (AUC={report["roc_auc"]:.4f})')
    ax[1].plot(rec, prec); ax[1].set(xlabel='Recall', ylabel='Precision',
                                       title=f'External PR (AP={report["average_precision"]:.4f})')
    fig.tight_layout(); fig.savefig(DEST / 'roc_pr.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    mat = np.array([[tn, fp], [fn, tp]])
    ax.imshow(mat, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Anomaly'],
           yticklabels=['Normal', 'Anomaly'], xlabel='Predicted', ylabel='Actual', title='External 200k confusion matrix')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{mat[r,c]:,}', ha='center', va='center', color='red')
    fig.tight_layout(); fig.savefig(DEST / 'confusion_matrix.png', dpi=160); plt.close(fig)

    old_report = json_read(RUN / 'evaluation/report.json')
    old_test = old_report['comparisons'][KIND]['test']
    print('\n===== 5-EPOCH ORIGINAL TEST versus NEW 200K HOLDOUT =====')
    for key in ('precision', 'recall', 'f1', 'average_precision', 'roc_auc'):
        old_value = float(old_test[key]); new_value = report[key]
        print(f'{key:18s} original={old_value:.6f}  external={new_value:.6f}  change_pp={(new_value-old_value)*100:+.3f}')
    print(f'TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}')
    print('Result folder:', DEST)
    print('EXTERNAL HOLDOUT COMPLETE: model not retrained or retuned.', flush=True)


if __name__ == '__main__':
    main()
