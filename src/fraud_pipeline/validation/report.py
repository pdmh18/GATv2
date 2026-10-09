"""Dataset distributions, graph diagnostics and explicit feature availability policy."""
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta

from .core import timestamp

FEATURE_POLICY = {
    'target_scope': 'all_transaction_types',
    'transaction_types': ['TRANSFER', 'CASH_IN', 'CASH_OUT', 'PAYMENT', 'DEBIT'],
    'prediction_events': ['TransferCreated', 'TransactionRequested'],
    'allowed_request_fields': ['amount', 'fee', 'currency', 'channel', 'transactionType'],
    'derived_features': ['hour', 'day_of_week', 'calendar_tags_from_known_calendar',
                         'prior_request_count', 'prior_completed_amount_stats', 'time_since_prior_request', 'new_receiver'],
    'join_only': ['event_id', 'transaction_id', 'sourceAccountId', 'destinationAccountId'],
    'excluded': ['risk', 'risk_score', 'is_anomaly', 'scenario_id', 'terminal_status', 'posted_at',
                 'sourceBalanceAfter', 'destinationBalanceAfter', 'loan_fields', 'customer_identity',
                 'snapshot_fields_without_availability_proof', 'generation_behavior_profiles'],
    'label_source': 'labels.jsonl; event_id points to request; join on transaction_id',
    'history_rule': 'Strictly prior events; no current terminal event or future data; no labels in graph/state updates.',
    'graph_policy': {
        'TRANSFER': 'Account source → account destination; created for request statistics, completed for settled flow.',
        'PAYMENT': 'Account → shared merchant; pending/failed edges must be distinguished from completed flow.',
        'DEBIT': 'Account → shared merchant; same availability rule as PAYMENT.',
        'CASH_OUT': 'Account → cash counterparty for money-flow graph.',
        'CASH_IN': 'Cash counterparty → credited account for money-flow graph; raw source field names the credited account.',
        'note': 'Keep raw source/destination roles in the transaction classification head. Phase 3 defines tensor direction, graph window and batching.'},
    'generation_profile_policy': 'behavior_profiles.jsonl is generator audit metadata; never feed oracle preferences or typical_amount into baseline/model.',
    'credit_policy': 'Excluded from initial features; dates checked for future data. Enable only after implementing as-of event lookup.',
    'loan_policy': 'Excluded: submission leaks outcome, snapshots describe final state and loan cash flows do not mutate transaction ledger.',
}


def distribution(rows, labels):
    amounts = sorted(r['amount'] for r in rows)
    count = len(rows)
    anomalies = sum(labels.get(r['transaction_id'], {}).get('is_anomaly') is True for r in rows)
    return {'count': count, 'normal': count - anomalies, 'anomaly': anomalies,
            'anomaly_rate': anomalies / count if count else None,
            'amount': {'min': amounts[0] if amounts else None, 'max': amounts[-1] if amounts else None,
                       'mean': sum(amounts) / count if count else None, 'sum': sum(amounts)},
            'type': dict(sorted(Counter(r['type'] for r in rows).items())),
            'channel': dict(sorted(Counter(r['channel'] for r in rows).items())),
            'status': dict(sorted(Counter(r['status'] for r in rows).items()))}


def summaries(transactions, labels, events, manifest):
    rows = list(transactions.values())
    daily, monthly = defaultdict(list), defaultdict(list)
    period = None
    try:
        start = timestamp(manifest['start_at'])
        end = datetime.combine(datetime.fromisoformat(manifest['end_date_exclusive']).date(), time(), start.tzinfo)
        if end <= start:
            raise ValueError('Empty simulation period')
        period = (start, end)
        d = start.date()
        while d < end.date():
            daily[d.isoformat()] = []
            monthly[d.isoformat()[:7]] = []
            d += timedelta(days=1)
    except (KeyError, ValueError, TypeError):
        pass
    for tx in rows:
        t = timestamp(tx['created_at'])
        # Normalize dates to simulation timezone, not string-prefix offsets.
        if period:
            t = t.astimezone(period[0].tzinfo)
        daily[t.date().isoformat()].append(tx)
        monthly[t.date().isoformat()[:7]].append(tx)
    pairs = Counter((r['source']['account_id'], r['destination']['account_id']) for r in rows)
    receiver = Counter(r['destination']['account_id'] for r in rows)
    all_nodes = {n for pair in pairs for n in pair}
    anomaly_amounts = [r['amount'] for r in rows if labels.get(r['transaction_id'], {}).get('is_anomaly') is True]
    normal_amounts = [r['amount'] for r in rows if labels.get(r['transaction_id'], {}).get('is_anomaly') is False]
    previews = []
    if period:
        start, end = period
        days = (end.date() - start.date()).days
        midnight = datetime.combine(start.date(), time(), start.tzinfo)
        first, second = midnight + timedelta(days=int(days * 0.5)), midnight + timedelta(days=int(days * 0.8))
        for name, lower, upper in [('train', start, first), ('validation', first, second), ('test', second, end)]:
            selected = [r for r in rows if lower <= timestamp(r['created_at']) < upper]
            previews.append({'name': name, 'from_inclusive': lower.isoformat(), 'to_exclusive': upper.isoformat(),
                             **distribution(selected, labels)})
    return {'overall': distribution(rows, labels),
            'by_type': {kind: distribution([r for r in rows if r['type'] == kind], labels) for kind in FEATURE_POLICY['transaction_types']},
            'daily': {k: distribution(v, labels) for k, v in sorted(daily.items())},
            'monthly': {k: distribution(v, labels) for k, v in sorted(monthly.items())},
            'event_types': dict(sorted(Counter(e['event_type'] for e in events).items())),
            'graph': {'observed_nodes': len(all_nodes), 'unique_pairs': len(pairs), 'unique_receivers': len(receiver),
                      'repeated_pairs': sum(n > 1 for n in pairs.values()), 'repeated_receivers': sum(n > 1 for n in receiver.values()),
                      'repeat_pair_transactions_after_first': sum(n - 1 for n in pairs.values())},
            'amount_range_overlap': bool(anomaly_amounts and normal_amounts and
                min(anomaly_amounts) <= max(normal_amounts) and min(normal_amounts) <= max(anomaly_amounts)),
            'time_split_preview': previews,
            'preview_note': 'Diagnostic only; no split artifacts created. Cutoffs use whole simulation days: floor(days*0.5), floor(days*0.8), at local midnight.'}


def markdown(report):
    summary = report['summary']
    lines = ['# Dataset validation — Phase 1', '',
             f"Status: **{report['status']}**; errors: {report['error_count']}; warnings: {report['warning_count']}.",
             f"Dataset: `{report['dataset']}`; schema: {report['manifest'].get('schema_version')}; seed: {report['manifest'].get('seed')}.", '',
             'Target: all five transaction types. Score request events before settlement.', '',
             '## Checks and findings', '', '| Severity | Code | Count | Example location | Meaning |', '|---|---|---:|---|---|']
    grouped = defaultdict(list)
    for issue in report['issues']:
        grouped[(issue['severity'], issue['code'], issue['message'])].append(issue['location'])
    for (severity, code, message), locations in sorted(grouped.items()):
        # Escape user-provided JSON content in Markdown cells.
        clean = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')
        lines.append(f'| {severity} | {clean(code)} | {len(locations)} | {clean(locations[0])} | {clean(message)} |')
    if not grouped:
        lines.append('| — | — | 0 | — | No findings |')
    lines += ['', '## Volume and graph', '', f"Transactions: {summary['overall']['count']}; anomalies: {summary['overall']['anomaly']}.",
              f"Graph diagnostics: `{summary['graph']}`.", f"Normal/anomaly amount ranges overlap: {summary['amount_range_overlap']}.", '',
              '## Per-type distribution', '', '| Type | Count | Normal | Anomaly | Amount min | Amount max |', '|---|---:|---:|---:|---:|---:|']
    for kind, data in summary['by_type'].items():
        lines.append(f"| {kind} | {data['count']} | {data['normal']} | {data['anomaly']} | {data['amount']['min']} | {data['amount']['max']} |")
    lines += ['', '## Time split preview (no split files)', '', '| Set | From inclusive | To exclusive | Count | Normal | Anomaly |', '|---|---|---|---:|---:|---:|']
    for data in summary['time_split_preview']:
        lines.append(f"| {data['name']} | {data['from_inclusive']} | {data['to_exclusive']} | {data['count']} | {data['normal']} | {data['anomaly']} |")
    lines += ['', '## Daily and monthly distributions', '', 'JSON includes type/channel/status, amount range/mean/total and label counts for every day (including zero-transaction days) and month.', '']
    for period in ['monthly', 'daily']:
        lines += [f'### {period}', '', '| Period | Count | Normal | Anomaly | Amount min | Amount max |', '|---|---:|---:|---:|---:|---:|']
        for key, data in summary[period].items():
            lines.append(f"| {key} | {data['count']} | {data['normal']} | {data['anomaly']} | {data['amount']['min']} | {data['amount']['max']} |")
        lines.append('')
    lines += ['## Feature availability', '',
              '- Use request amount/fee/currency/channel/type and causal history. Account IDs are join/mapping keys.',
              '- Exclude ground truth, terminal status, posted_at, after-balances, loan fields and identity data.',
              '- Credit history is checked but excluded until as-of lookup exists. Loan submission already includes final outcome; loan events do not update the transaction ledger.',
              '- CASH_IN cash-flow direction is counterparty → credited account; raw source field identifies the credited account. Other flows are source → destination.',
              '- Native transfer export duplicates the transfer subset of events.jsonl; never concatenate both as independent events.',
              '- Passed-with-warnings is dataset contract validation, not real backend replay or proof of model accuracy.', '',
              '## Provenance', '', 'SHA256 for manifest and every input file is recorded in validation.json. No dataset is modified by this command.', '']
    return '\n'.join(lines)
