"""Stream request-time examples; terminal snapshots never enter sample features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..validation.contracts import REQUESTS, validate_event
from ..validation.core import Audit, text, timestamp


class DatasetError(ValueError):
    pass


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_validation(source: Path, report_path: Path):
    """Reuse a successful Phase 1 audit only for the exact same file contents."""
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise DatasetError(f'Cannot read Phase 1 report: {report_path}; run scripts/validate_dataset.py first') from exc
    if not isinstance(report, dict):
        raise DatasetError('Phase 1 report must be a JSON object')
    if report.get('status') not in {'passed', 'passed_with_warnings'} or report.get('error_count') != 0:
        raise DatasetError('Phase 1 validation failed; repair dataset and rerun validator before splitting')
    if Path(report.get('dataset', '')).resolve() != source.resolve():
        raise DatasetError('Validation report belongs to a different dataset directory')
    try:
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        expected = {'manifest.json', *(name + '.jsonl' for name in manifest['counts'])}
        hashes = report['sha256']
        if set(hashes) != expected or report['manifest'] != manifest:
            raise DatasetError('Validation report and source manifest disagree')
        for name in sorted(expected):
            # Dataset table names cannot select paths outside source.
            if Path(name).name != name or file_hash(source / name) != hashes[name]:
                raise DatasetError(f'Stale validation report: {name} changed; rerun validator')
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError('Invalid validation provenance or unreadable source files') from exc
    return manifest, hashes, report


def jsonl_rows(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected object')
            except ValueError as exc:
                raise DatasetError(f'{path.name}:{line_no}: invalid JSON object') from exc
            yield line_no, row


def load_labels(source: Path):
    labels, event_ids = {}, set()
    for line_no, row in jsonl_rows(source / 'labels.jsonl'):
        tx_id, event_id, target = row.get('transaction_id'), row.get('event_id'), row.get('is_anomaly')
        if not text(tx_id) or not text(event_id) or type(target) is not bool:
            raise DatasetError(f'labels:{line_no}: invalid ID/target')
        if tx_id in labels or event_id in event_ids:
            raise DatasetError(f'labels:{line_no}: duplicate transaction/event ID')
        labels[tx_id] = (event_id, int(target))
        event_ids.add(event_id)
    return labels


def request_samples(source: Path, timezone):
    """Yield one joined sample per request, in original chronological file order."""
    labels = load_labels(source)
    remaining = set(labels)
    previous = None
    for line_no, event in jsonl_rows(source / 'events.jsonl'):
        try:
            occurred = timestamp(event.get('occurred_at'))
        except (ValueError, TypeError) as exc:
            raise DatasetError(f'events:{line_no}: invalid timezone-aware timestamp') from exc
        if not text(event.get('event_type')):
            raise DatasetError(f'events:{line_no}: invalid event type')
        if previous is not None and occurred < previous:
            raise DatasetError(f'events:{line_no}: nonchronological event stream')
        previous = occurred
        if event.get('event_type') not in REQUESTS:
            continue
        audit = Audit()
        if not validate_event(audit, event, f'events:{line_no}'):
            raise DatasetError(f'events:{line_no}: invalid request: {audit.issues}')
        tx_id = event['aggregate_id']
        if tx_id not in remaining:
            raise DatasetError(f'events:{line_no}: missing label or duplicate request for {tx_id}')
        label_event, target = labels[tx_id]
        if label_event != event['event_id']:
            raise DatasetError(f'events:{line_no}: label must point to this request event')
        remaining.remove(tx_id)
        payload = event['payload']
        yield {
            'event_id': event['event_id'], 'transaction_id': tx_id,
            'event_type': event['event_type'], 'source_event_line': line_no,
            'timestamp': occurred.astimezone(timezone).isoformat(),
            'src_acc': payload['sourceAccountId'], 'dst_acc': payload['destinationAccountId'],
            'amount': payload['amount'], 'fee': payload['fee'], 'currency': payload['currency'],
            'type': 'TRANSFER' if event['event_type'] == 'TransferCreated' else payload['transactionType'],
            'channel': payload['channel'], 'label': target,
        }
    if remaining:
        raise DatasetError(f'{len(remaining)} labels have no request event')
