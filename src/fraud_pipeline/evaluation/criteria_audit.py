"""Criteria source readiness and descriptive EDA; never manufacture criterion matches/labels."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..features.causal import REQUESTS, COMPLETED, microseconds
from ..ingestion.request_loader import file_hash, jsonl_rows, verify_validation

KINDS = {'behavior_signal', 'external_fact', 'contextual_review'}
MODES = {'request_or_monitor', 'settlement_or_monitor', 'end_of_day', 'periodic_review', 'verification'}
UNITS = {'transaction', 'account_window', 'device_window', 'entity_group', 'customer_or_account'}
CAPABILITIES = {'request_history', 'settled_history', 'directed_timeline', 'balance_snapshots',
                'device_history', 'login_history', 'security_changes', 'geo_history', 'watchlist_asof',
                'identity_checks', 'observed_profile', 'contact_checks', 'purpose_evidence',
                'reward_context', 'refund_history', 'terminal_history', 'account_lifecycle'}


def load_catalog(path):
    catalog = json.loads(Path(path).read_text())
    expected = {f'{group}{i:02d}' for group, count in zip('ABCDEFGH', (4, 6, 6, 8, 6, 7, 7, 5))
                for i in range(1, count+1)}
    rows = catalog['criteria']
    if catalog['schema_version'] != 2 or len(rows) != 49 or {r['id'] for r in rows} != expected:
        raise ValueError('Criteria catalog must retain the 49 unique version-2 IDs')
    for row in rows:
        if (row['kind'] not in KINDS or row['evaluation_mode'] not in MODES or row['unit'] not in UNITS
                or not row['requires'] or set(row['requires'])-CAPABILITIES
                or type(row['minimum_calendar_history_days']) is not int or row['minimum_calendar_history_days'] < 0
                or not row['formula'] or not row['normal_counterexample']
                or row['outcome'] != 'matched/not_matched/insufficient_data'):
            raise ValueError(f'Incomplete criteria contract: {row["id"]}')
    return catalog


def render_catalog(catalog):
    modes = {'request_or_monitor': 'Request / giám sát', 'settlement_or_monitor': 'Sau settled / giám sát',
             'end_of_day': 'Cuối ngày', 'periodic_review': 'Rà soát định kỳ', 'verification': 'Xác minh'}
    groups = dict(zip('ABCDEFGH', ('Giá trị', 'Tần suất', 'Luân chuyển tiền', 'Đối tác / mạng lưới',
                                 'Số dư / dòng tiền', 'Ngữ cảnh / thiết bị', 'Định danh / hồ sơ', 'Mẫu hình / mục đích')))
    lines = []
    for group, title in groups.items():
        lines += [f'### {group}. {title}', '', '| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |',
                  '| --- | --- | --- | --- |']
        for row in catalog['criteria']:
            if row['id'][0] != group:
                continue
            # Escape pipe characters so the generated table cannot lose columns.
            cells = [row['id'], row['name']+'; '+row['unit'], modes[row['evaluation_mode']]+'; '+row['kind'], row['formula']]
            lines.append('| '+' | '.join(c.replace('|', '\\|').replace('\n', ' ') for c in cells)+' |')
        lines.append('')
        for row in catalog['criteria']:
            if row['id'][0] != group:
                continue
            params = json.dumps(row['parameters'], ensure_ascii=False)
            lines += [f"- **{row['id']}** — nguồn: `{', '.join(row['requires'])}`; lịch tối thiểu: {row['minimum_calendar_history_days']} ngày; tham số: `{params}`. "
                      f"Trường hợp hợp lệ dễ nhầm: {row['normal_counterexample']} Hỗ trợ hiện tại: {row['existing_model_support']}"]
        lines.append('')
    return '\n'.join(lines)


def readiness_rows(catalog, capabilities, timestamps, masks, start_us):
    rows = []
    for criterion in catalog['criteria']:
        missing = sorted(set(criterion['requires'])-set(capabilities))
        present = sorted(set(criterion['requires']) & set(capabilities))
        state = 'sources_present' if not missing else ('partial_sources' if present else 'missing_sources')
        days = criterion['minimum_calendar_history_days']
        row = {'id': criterion['id'], 'kind': criterion['kind'], 'unit': criterion['unit'],
               'mode': criterion['evaluation_mode'], 'source_readiness': state,
               'present_sources': ','.join(present), 'missing_sources': ','.join(missing),
               'unconfigured_parameters': ','.join(k for k, v in criterion['parameters'].items() if v is None),
               'minimum_calendar_history_days': days,
               'criterion_matching': 'not_implemented_by_this_eda',
               'existing_model_support': criterion['existing_model_support']}
        for split, mask in masks.items():
            selected = timestamps[mask]
            # Necessary calendar condition only: not an assertion of per-account support/completeness.
            row[split+'_calendar_upper_bound'] = float(np.mean(selected >= start_us+days*86400*1000000)) if len(selected) else None
        rows.append(row)
    return rows


def summarize_events(events):
    counts, fields, platforms = Counter(), Counter(), Counter()
    devices = defaultdict(set)
    requests = 0
    for event in events:
        kind = event['event_type']
        counts[kind] += 1
        if kind not in REQUESTS:
            continue
        requests += 1
        p = event['payload']
        for key in ('device_id', 'platform', 'authentication_method', 'authentication_verified'):
            fields[key] += p.get(key) is not None
        if p.get('platform') is not None:
            platforms[p['platform'].lower()] += 1
        if p.get('device_id') is not None:
            devices[p['device_id']].add(p['sourceAccountId'])
    capabilities = set()
    if requests:
        capabilities.update(('request_history', 'directed_timeline'))
    if any(counts[k] for k in COMPLETED):
        capabilities.add('settled_history')
    if counts['AccountOpened']:
        capabilities.add('account_lifecycle')
    if any(counts[k] for k in ('TransferCompleted', 'TransferCancelled', 'TransactionPosted', 'TransactionRejected')):
        capabilities.add('terminal_history')
    if fields['device_id']:
        capabilities.add('device_history')
    # External sources are deliberately not inferred from generated profile names or terminal auth.
    result = {'event_counts': dict(counts), 'request_count': requests,
              'request_field_coverage': {k: {'present': fields[k], 'fraction': fields[k]/requests if requests else 0.}
                                         for k in ('device_id', 'platform', 'authentication_method', 'authentication_verified')},
              'platform_counts': dict(platforms), 'distinct_devices': len(devices),
              'devices_shared_across_accounts': sum(len(accounts)>1 for accounts in devices.values()),
              'max_accounts_per_device': max(map(len, devices.values()), default=0)}
    return capabilities, result


def run(source, validation, dataset, catalog_path, output):
    source, validation, output, catalog_path = map(Path, (source, validation, output, catalog_path))
    if output.exists():
        raise ValueError('Criteria EDA output exists; choose a new report version')
    manifest, hashes, _ = verify_validation(source, validation)
    if hashes != dataset.manifest['source_sha256']:
        raise ValueError('Criteria EDA raw source differs from feature artifact')
    catalog = load_catalog(catalog_path)
    print('Scanning observable event fields and device sharing', flush=True)
    capabilities, observable = summarize_events(row for _, row in jsonl_rows(source/'events.jsonl'))
    # Existence of balance snapshots is distinct from a complete all-product ledger.
    for _, event in jsonl_rows(source/'events.jsonl'):
        if event['event_type'] in COMPLETED and all(k in event['payload'] for k in ('sourceBalanceBefore', 'sourceBalanceAfter')):
            capabilities.add('balance_snapshots')
            break
    a = dataset.arrays
    masks = {split: np.asarray(a[split+'_mask']) for split in ('train', 'validation', 'test')}
    rows = readiness_rows(catalog, capabilities, a['request_timestamps_us'], masks, microseconds(manifest['start_at']))
    names = dataset.manifest['feature_names']
    descriptive = {}
    for split, mask in masks.items():
        x = a['transaction_features'][mask]
        def fraction(name):
            return float(x[:, names.index(name)].mean()) if name in names else None
        descriptive[split] = {'count': int(mask.sum()), 'cold_start_fraction': fraction('cold_start'),
                              'median_reference_reliable_fraction': fraction('type_amount_reference_reliable'),
                              'mean_reference_reliable_fraction': fraction('type_mean_reference_reliable'),
                              'new_device_fraction': fraction('new_device'), 'new_receiver_fraction': fraction('new_receiver')}
    # Labels used only for descriptive TRAIN cohorts, never to define a match or tune a threshold.
    train = masks['train']
    cohorts = {}
    for label, name in ((0, 'normal'), (1, 'anomaly')):
        selected = train & (a['labels'] == label)
        cohorts[name] = {'count': int(selected.sum())}
        for feature in ('new_device', 'new_receiver'):
            cohorts[name][feature+'_fraction'] = float(a['transaction_features'][selected, names.index(feature)].mean())
    report = {'status': 'complete', 'catalog_sha256': file_hash(catalog_path),
              'feature_manifest_sha256': file_hash(dataset.directory/'manifest.json'), 'dataset_sha256': hashes,
              'catalog_version': catalog['profile'], 'capabilities': sorted(capabilities), 'observable': observable,
              'readiness_counts': dict(Counter(row['source_readiness'] for row in rows)),
              'criteria': rows, 'feature_coverage': descriptive, 'train_only_label_cohorts': cohorts,
              'limits': ['Source readiness is not an implemented rule, satisfied criterion, or bank-data validation.',
                         'Calendar support is an upper bound; per-account count, zero denominators and watermark may still make evaluation impossible.',
                         'No criterion matches, anomaly labels, model predictions or threshold tuning are generated.',
                         'Test values are descriptive coverage only; no feature/threshold selection from test.',
                         'Balance snapshots cover the synthetic transaction ledger; loan/funding completeness is not established.',
                         'Device IDs are synthetic and per-account; zero shared devices is generator coverage, not a real-bank finding.']}
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (output/'criteria_readiness.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Criteria v2 — dataset readiness EDA', '',
             'Descriptive audit of the current generated dataset. No rule matching or relabeling.', '',
             '| Criterion | Source readiness | Missing sources | Unconfigured thresholds | Train calendar upper bound |',
             '|---|---|---|---|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['source_readiness']} | {row['missing_sources']} | {row['unconfigured_parameters']} | {row['train_calendar_upper_bound']:.2%} |")
    lines += ['', '## Observable context', '',
              f"Requests: {observable['request_count']:,}; devices: {observable['distinct_devices']:,}; shared devices: {observable['devices_shared_across_accounts']:,}.", '',
              '| Field | Present fraction |', '|---|---:|']
    for name, values in observable['request_field_coverage'].items():
        lines.append(f"| {name} | {values['fraction']:.2%} |")
    lines += ['', '## Limits', ''] + ['- '+s for s in report['limits']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    plot(report, output)
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = {'sources_present': '#059669', 'partial_sources': '#d97706', 'missing_sources': '#dc2626'}
    with PdfPages(output/'criteria_readiness.pdf') as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = report['readiness_counts']
        axes[0].bar(list(counts), list(counts.values()), color=[colors[k] for k in counts])
        axes[0].set(title='Source readiness (not rule implementation)', ylabel='Number of criteria')
        axes[0].tick_params(axis='x', rotation=15)
        coverage = report['observable']['request_field_coverage']
        axes[1].bar(list(coverage), [v['fraction']*100 for v in coverage.values()], color='#2563eb')
        axes[1].set(title='Observable request metadata', ylabel='Requests with field (%)', ylim=(0, 105))
        axes[1].tick_params(axis='x', rotation=25)
        fig.tight_layout(); fig.savefig(output/'source_readiness.png', dpi=160); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5))
        days = sorted({c['minimum_calendar_history_days'] for c in report['criteria'] if c['minimum_calendar_history_days']})
        for split in ('train', 'validation', 'test'):
            values = [next(c[split+'_calendar_upper_bound'] for c in report['criteria'] if c['minimum_calendar_history_days']==day) for day in days]
            ax.plot(days, np.asarray(values)*100, marker='o', label=split)
        ax.set(xlabel='Required prior calendar days', ylabel='Eligible requests, upper bound (%)', ylim=(-2, 102),
               title='Calendar coverage only — not per-account/reference support')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(output/'history_coverage.png', dpi=160); pdf.savefig(fig); plt.close(fig)
