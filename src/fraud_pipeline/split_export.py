"""Phase 2: deterministic streaming exports with source and output provenance."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from .ingestion.request_loader import DatasetError, file_hash, request_samples, verify_validation
from .validation.core import timestamp


class Counts:
    def __init__(self):
        self.count = self.normal = self.anomaly = self.total = 0
        self.minimum = self.maximum = None
        self.first = self.last = None
        self.types, self.channels, self.months = Counter(), Counter(), defaultdict(lambda: [0, 0, 0])

    def add(self, row):
        amount, label = row['amount'], row['label']
        self.count += 1
        self.normal += label == 0
        self.anomaly += label == 1
        self.total += amount
        self.minimum = amount if self.minimum is None else min(self.minimum, amount)
        self.maximum = amount if self.maximum is None else max(self.maximum, amount)
        self.first = self.first or row['timestamp']
        self.last = row['timestamp']
        self.types[row['type']] += 1
        self.channels[row['channel']] += 1
        month = self.months[row['timestamp'][:7]]
        month[0] += 1
        month[1 + label] += 1

    def result(self):
        return {'count': self.count, 'normal': self.normal, 'anomaly': self.anomaly,
                'anomaly_rate': self.anomaly / self.count if self.count else None,
                'first_request_at': self.first, 'last_request_at': self.last,
                'amount': {'min': self.minimum, 'max': self.maximum, 'sum': self.total,
                           'mean': self.total / self.count if self.count else None},
                'type': dict(sorted(self.types.items())), 'channel': dict(sorted(self.channels.items())),
                'monthly': {k: dict(count=v[0], normal=v[1], anomaly=v[2]) for k, v in sorted(self.months.items())}}


def split_intervals(manifest, ratios):
    if ratios.get('time_based') is not True:
        raise DatasetError('Phase 2 requires time_based=true')
    try:
        values = [Decimal(str(ratios[k])) for k in ('train', 'val', 'test')]
        if any(not v.is_finite() or not 0 < v < 1 for v in values) or sum(values) != 1:
            raise ValueError('Ratios must be positive and sum to 1')
        start = timestamp(manifest['start_at'])
        days = manifest['days']
        if type(days) is not int or days < 3:
            raise ValueError('At least three simulation days are needed')
        midnight = datetime.combine(start.date(), time(), start.tzinfo)
        end = datetime.combine(datetime.fromisoformat(manifest['end_date_exclusive']).date(), time(), start.tzinfo)
        if end != midnight + timedelta(days=days):
            raise ValueError('Manifest dates and days disagree')
        train_days, before_test = int(days * values[0]), int(days * (values[0] + values[1]))
        if not 0 < train_days < before_test < days:
            raise ValueError('Each set needs at least one whole day')
        cuts = [start, midnight + timedelta(days=train_days), midnight + timedelta(days=before_test), end]
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise DatasetError(f'Invalid chronological split configuration: {exc}') from exc
    return {name: (cuts[i], cuts[i+1]) for i, name in enumerate(('train', 'validation', 'test'))}


def export_splits(source, validation_report, output, ratios, *, overwrite=False):
    source, validation_report, output = Path(source).resolve(), Path(validation_report).resolve(), Path(output).resolve()
    # Keep source and audit outside the directory that can be replaced.
    if output == source or output in source.parents or source in output.parents or output in validation_report.parents:
        raise DatasetError('Output must be separate from source dataset and validation report')
    if output.exists() and not overwrite:
        raise DatasetError(f'Output already exists: {output}; use --overwrite or another output path')
    manifest, source_hashes, audit = verify_validation(source, validation_report)
    intervals = split_intervals(manifest, ratios)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.split-build-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'dataset'
        staged.mkdir()
        counts = {name: Counts() for name in intervals}
        digests = {name: hashlib.sha256() for name in intervals}
        handles = {name: (staged / f'{name}.jsonl').open('wb') for name in intervals}
        try:
            for row in request_samples(source, next(iter(intervals.values()))[0].tzinfo):
                occurred = timestamp(row['timestamp'])
                name = next((name for name, (lower, upper) in intervals.items() if lower <= occurred < upper), None)
                if name is None:
                    raise DatasetError(f"Request outside simulation period: {row['event_id']}")
                encoded = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
                handles[name].write(encoded)
                digests[name].update(encoded)
                counts[name].add(row)
        finally:
            for handle in handles.values():
                handle.close()
        total = sum(c.count for c in counts.values())
        if total != manifest['counts']['transactions'] or total != manifest['counts']['labels']:
            raise DatasetError('Request/label count differs from dataset manifest')
        if any(c.count == 0 for c in counts.values()):
            raise DatasetError('A chronological split is empty; increase dataset size or revise generation period')
        warnings = [f'{name} lacks a class; adjust generator before evaluation, do not resample held-out data'
                    for name, c in counts.items() if not c.normal or not c.anomaly]
        # Guard against source replacement/modification during export, before promotion.
        for name, expected in source_hashes.items():
            if file_hash(source / name) != expected:
                raise DatasetError(f'Source changed during export: {name}; rerun validator')
        result = {
            'schema_version': 1, 'dataset_schema_version': manifest['schema_version'],
            'source': {'directory': str(source), 'seed': manifest['seed'], 'source_profile': manifest['source_profile'],
                       'manifest': manifest, 'sha256': source_hashes},
            'validation': {'status': audit['status'], 'error_count': audit['error_count'],
                           'warning_count': audit['warning_count'], 'report_sha256': file_hash(validation_report)},
            'target_scope': 'all_transaction_types', 'prediction_events': ['TransferCreated', 'TransactionRequested'],
            'sample_fields': ['event_id', 'transaction_id', 'event_type', 'source_event_line', 'timestamp',
                              'src_acc', 'dst_acc', 'amount', 'fee', 'currency', 'type', 'channel', 'label'],
            'trace': 'source.directory/events.jsonl at source_event_line (1-based); verify source SHA256 before replay',
            'target': {'field': 'label', 'normal': 0, 'anomaly': 1, 'excluded_from_features': True},
            'excluded_features': ['terminal_status', 'posted_at', 'after_balances', 'risk', 'scenario_id', 'credit', 'loan', 'identity'],
            'split_rule': {'ratios': ratios, 'unit': 'whole_simulation_days', 'interval': '[from_inclusive,to_exclusive)',
                           'cutoffs': 'floor(days*train), floor(days*(train+val)), local midnight',
                           'resampling': False, 'same_timestamp_stays_together': True},
            'total_samples': total, 'warnings': warnings,
            'splits': {name: {'file': f'{name}.jsonl', 'sha256': digests[name].hexdigest(),
                       'from_inclusive': lower.isoformat(), 'to_exclusive': upper.isoformat(), **counts[name].result()}
                       for name, (lower, upper) in intervals.items()},
            'history': 'Phase 3 must replay source events in time order; train/validation/test files contain request samples only.'}
        (staged / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (staged / 'summary.md').write_text(split_summary(result), encoding='utf-8')
        backup = None
        if output.exists():
            if not output.is_dir() or not (output / 'manifest.json').exists():
                raise DatasetError('Refusing to replace directory without a split manifest')
            backup = output.with_name(output.name + '.previous-' + uuid4().hex[:8])
            output.rename(backup)
        try:
            os.replace(staged, output)
        except OSError:
            if backup is not None:
                backup.rename(output)
            raise
    return result


def split_summary(manifest):
    lines = ['# Phase 2 — request dataset splits', '',
             f"Source seed: {manifest['source']['seed']}; dataset schema: {manifest['dataset_schema_version']}.",
             f"Phase 1: {manifest['validation']['status']}; total samples: {manifest['total_samples']}.", '',
             '| Set | From inclusive | To exclusive | Count | Normal | Anomaly |', '|---|---|---|---:|---:|---:|']
    for name, data in manifest['splits'].items():
        lines.append(f"| {name} | {data['from_inclusive']} | {data['to_exclusive']} | {data['count']} | {data['normal']} | {data['anomaly']} |")
    lines += ['', 'All five transaction types; one row per request. No resampling or preprocessing fit.',
              'label is the target, IDs/line numbers are trace keys, not model features.',
              'Phase 3 needs the original chronological event stream for history and settled graph updates.', '']
    lines.extend('- ' + warning for warning in manifest['warnings'])
    return '\n'.join(lines) + '\n'
