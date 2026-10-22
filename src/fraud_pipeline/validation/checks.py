"""Cross-file integrity, transaction lifecycle and chronological balance replay."""
from collections import defaultdict
from datetime import date
from uuid import NAMESPACE_OID, uuid5

from .contracts import CHANNELS, REQUESTS, TERMINALS, TRANSFER_TYPES, TYPES, validate_event
from .core import fields, index_rows, integer, no_leakage, text, timestamp, time_value


def validate_tables(tables, manifest, audit):
    customers = index_rows(tables['customers'], '_id', audit, 'customers')
    accounts = index_rows(tables['accounts'], '_id', audit, 'accounts')
    partners = index_rows(tables['counterparties'], '_id', audit, 'counterparties')
    transactions = index_rows(tables['transactions'], 'transaction_id', audit, 'transactions')
    labels = index_rows(tables['labels'], 'transaction_id', audit, 'labels')
    events = index_rows(tables['events'], 'event_id', audit, 'events')
    native = index_rows(tables['transfer_events'], 'eventId', audit, 'transfer_events')
    index_rows(tables['labels'], 'event_id', audit, 'labels.event_id')
    index_rows(tables['transactions'], 'idempotency_key', audit, 'transactions.idempotency_key')
    index_rows(tables['accounts'], 'account_no', audit, 'accounts.account_no')
    audit.check(not (accounts.keys() & partners.keys()), 'node_collision', 'counterparties', 'Account and counterparty IDs overlap')
    applications = index_rows(tables['loan_applications'], '_id', audit, 'loan_applications')
    packages = index_rows(tables['loan_packages'], '_id', audit, 'loan_packages')
    loans = index_rows(tables['loans'], 'loan_no', audit, 'loans')
    for identity, app in applications.items():
        audit.check(text(app.get('customer_id')) and text(app.get('loan_package_id')) and app['customer_id'] in customers and app['loan_package_id'] in packages,
                    'reference', identity, 'Loan application references unknown customer/package')
    for identity, loan in loans.items():
        audit.check(all(text(loan.get(k)) for k in ('application_id', 'customer_id', 'disbursement_account_id')) and loan['application_id'] in applications and loan['customer_id'] in customers
                    and loan['disbursement_account_id'] in accounts, 'reference', identity, 'Loan snapshot references unknown application/customer/account')
    valid_accounts = {}
    for identity, account in accounts.items():
        loc = f'account:{identity}'
        ok = fields(audit, account, {'customer_id': text, 'balance': lambda v: isinstance(v, dict),
            'created_at': time_value, 'currency': lambda v: v == 'VND'}, loc)
        if ok:
            ok = fields(audit, account['balance'], {'current': integer}, loc + '.balance')
            audit.check(account['customer_id'] in customers, 'reference', loc, 'Unknown customer')
            if ok:
                valid_accounts[identity] = account
    for identity, partner in partners.items():
        fields(audit, partner, {'_id': text, 'type': lambda v: v in {'merchant', 'cash'}, 'created_at': time_value},
               f'counterparty:{identity}', exact=True)
    valid_transactions = {}
    for identity, tx in transactions.items():
        loc = f'transaction:{identity}'
        ok = fields(audit, tx, {'transaction_id': text, 'source': lambda v: isinstance(v, dict),
            'destination': lambda v: isinstance(v, dict), 'amount': lambda v: integer(v, 1), 'fee': integer,
            'currency': lambda v: v == 'VND', 'type': lambda v: v in TYPES, 'channel': lambda v: v in CHANNELS,
            'status': lambda v: v in {'success', 'failed'}, 'idempotency_key': text, 'created_at': time_value,
            'posted_at': lambda v: v is None or time_value(v), 'calendar_context': lambda v: isinstance(v, dict),
            **({'device': lambda v: v is None or (isinstance(v,dict) and set(v)=={'device_id','platform'} and text(v['device_id']) and text(v['platform']))} if 'device' in tx else {})}, loc, exact=True)
        audit.check(no_leakage(tx), 'leakage', loc, 'Risk/ground-truth must not appear in transaction snapshot')
        if not ok:
            continue
        ok = fields(audit, tx['source'], {'account_id': text, 'customer_id': text}, loc + '.source', exact=True)
        ok = fields(audit, tx['destination'], {'account_id': text, 'type': lambda v: v in {'account', 'merchant', 'cash'}}, loc + '.destination', exact=True) and ok
        fields(audit, tx['calendar_context'], {'tags': lambda v: isinstance(v, list) and all(text(t) for t in v),
            'campaign': lambda v: type(v) is bool, 'holiday': lambda v: type(v) is bool}, loc + '.calendar_context', exact=True)
        if not ok:
            continue
        source, dest = tx['source']['account_id'], tx['destination']['account_id']
        audit.check(source in valid_accounts, 'reference', loc, 'Unknown source account')
        if source in valid_accounts:
            audit.check(tx['source']['customer_id'] == valid_accounts[source]['customer_id'], 'reference', loc, 'Source customer mismatch')
        expected_kind = 'account' if tx['type'] == 'TRANSFER' else 'cash' if tx['type'] in {'CASH_IN', 'CASH_OUT'} else 'merchant'
        audit.check(tx['destination']['type'] == expected_kind, 'node_type', loc, 'Destination type disagrees with transaction type')
        pool = accounts if expected_kind == 'account' else partners
        audit.check(dest in pool and (expected_kind == 'account' or pool[dest].get('type') == expected_kind), 'reference', loc, 'Unknown or incorrectly typed destination')
        audit.check(source != dest, 'self_transfer', loc, 'Source and destination must differ')
        audit.check((tx['posted_at'] is None) == (tx['status'] == 'failed'), 'posted_at', loc, 'Failed transaction must have null posted_at; successful transaction must have timestamp')
        if tx['posted_at']:
            audit.check(timestamp(tx['posted_at']) >= timestamp(tx['created_at']), 'posted_at', loc, 'Posted before request')
        valid_transactions[identity] = tx
    audit.check(labels.keys() == transactions.keys(), 'label_join', 'labels', 'Labels must join transactions one-to-one without extra/missing IDs')
    for identity, label in labels.items():
        ok = fields(audit, label, {'event_id': text, 'transaction_id': text, 'is_anomaly': lambda v: type(v) is bool,
            'scenario_id': text, 'source_profile': text}, f'label:{identity}', exact=True)
        if ok:
            audit.check(label['source_profile'] == manifest.get('source_profile'), 'label_profile', identity, 'Source profile mismatch')
            audit.check((label['scenario_id'] == 'normal') == (not label['is_anomaly']), 'label_scenario', identity, 'Normal scenario and target disagree')
    valid_events = []
    groups = defaultdict(list)
    previous_time = None
    for n, event in enumerate(tables['events'], 1):
        loc = f'events:{n}'
        if not validate_event(audit, event, loc):
            continue
        occurred = timestamp(event['occurred_at'])
        audit.check(previous_time is None or occurred >= previous_time, 'event_order', loc, 'Event file is not chronological')
        previous_time = occurred
        valid_events.append(event)
        groups[(event['aggregate_type'], event['aggregate_id'])].append(event)
        if event['event_type'] in REQUESTS | TERMINALS:
            expected = str(uuid5(NAMESPACE_OID, f"{event['aggregate_id']}:{event['event_type']}"))
            audit.check(event['event_id'] == expected, 'event_id', loc, 'Transaction event ID differs from Anomaly algorithm')
    for (kind, identity), group in groups.items():
        seq = [e['sequence'] for e in group]
        audit.check(seq == list(range(1, len(group) + 1)), 'sequence', f'{kind}:{identity}', 'Sequence must be contiguous in file order')
    transaction_groups = {identity: group for (kind, identity), group in groups.items() if kind == 'transaction'}
    audit.check(transaction_groups.keys() == transactions.keys(), 'event_join', 'events', 'Transaction aggregates do not match transaction IDs')
    for identity, tx in valid_transactions.items():
        group = transaction_groups.get(identity, [])
        transfer = tx['type'] == 'TRANSFER'
        request_type = 'TransferCreated' if transfer else 'TransactionRequested'
        terminal = ('TransferCompleted' if transfer else 'TransactionPosted') if tx['status'] == 'success' else ('TransferCancelled' if transfer else 'TransactionRejected')
        if not audit.check([e['event_type'] for e in group] == [request_type, terminal], 'lifecycle', identity, 'Expected exactly request then matching result'):
            continue
        first, last = group
        audit.check(timestamp(first['occurred_at']) == timestamp(tx['created_at']), 'request_time', identity, 'Request and created_at mismatch')
        if tx['posted_at']:
            audit.check(timestamp(last['occurred_at']) == timestamp(tx['posted_at']), 'terminal_time', identity, 'Terminal and posted_at mismatch')
        for e in group:
            p = e['payload']
            expected = {'sourceAccountId': tx['source']['account_id'], 'destinationAccountId': tx['destination']['account_id'],
                        'amount': tx['amount'], 'fee': tx['fee'], 'currency': tx['currency'], 'channel': tx['channel']}
            if not transfer:
                expected['transactionType'] = tx['type']
            audit.check(all(p.get(k) == v for k, v in expected.items()), 'snapshot_match', e['event_id'], 'Event and transaction snapshot disagree')
        if isinstance(tx.get('device'),dict):
            for key in ('device_id','platform'):
                audit.check(first['payload'].get(key)==tx['device'][key], 'device_snapshot_match',identity,'Request device disagrees with transaction snapshot')
        label = labels.get(identity, {})
        audit.check(label.get('event_id') == first['event_id'], 'label_request', identity, 'Label must refer to request, not terminal event')
    for n, row in enumerate(tables['transfer_events'], 1):
        spec = {'eventId': text, 'transactionId': text, 'eventType': lambda v: v in TRANSFER_TYPES,
                'schemaVersion': lambda v: type(v) is int and v == 1, 'sequence': lambda v: integer(v, 1),
                'occurredAt': time_value, 'payload': lambda v: isinstance(v, dict)}
        if fields(audit, row, spec, f'transfer_events:{n}', exact=True):
            validate_event(audit, {'event_id': row['eventId'], 'aggregate_id': row['transactionId'],
                'aggregate_type': 'transaction', 'event_type': row['eventType'], 'schema_version': 2,
                'sequence': row['sequence'], 'occurred_at': row['occurredAt'], 'payload': row['payload'],
                'labels': None}, f'transfer_events:{n}')
    expected_native = {e['event_id']: {'eventId': e['event_id'], 'transactionId': e['aggregate_id'], 'eventType': e['event_type'],
        'schemaVersion': 1, 'sequence': e['sequence'], 'occurredAt': e['occurred_at'], 'payload': e['payload']}
        for e in valid_events if e['event_type'] in TRANSFER_TYPES}
    audit.check(native == expected_native, 'native_export', 'transfer_events', 'Native transfer export must exactly match canonical dataset transfer events')
    native_order = [(r.get('occurredAt'), r.get('transactionId'), r.get('sequence')) for r in tables['transfer_events']]
    try:
        audit.check(native_order == sorted(native_order), 'native_order', 'transfer_events', 'Native export not ordered')
    except TypeError:
        audit.check(False, 'native_order', 'transfer_events', 'Invalid native export sort fields')
    ledger = replay_ledger(valid_events, valid_accounts, valid_transactions, audit)
    check_history(tables, valid_events, customers, accounts, audit)
    return valid_transactions, labels, valid_events, ledger


def replay_ledger(events, accounts, transactions, audit):
    balances = {key: value['balance']['current'] for key, value in accounts.items()}
    seen_open = set()
    pending = {}
    replayed, fees = 0, 0
    for e in events:
        kind, p, identity = e['event_type'], e['payload'], e['aggregate_id']
        if kind == 'AccountOpened':
            aid = p['account_id']
            audit.check(aid == identity and aid in accounts, 'account_open', e['event_id'], 'Unknown account aggregate')
            audit.check(aid not in seen_open, 'account_open', aid, 'Account opened twice')
            if aid in accounts:
                a = accounts[aid]
                audit.check(p['opening_balance'] == a['balance']['current'] and p['customer_id'] == a['customer_id']
                    and p['account_no'] == a['account_no'] and timestamp(e['occurred_at']) == timestamp(a['created_at']),
                    'account_open', aid, 'Opening event and account snapshot disagree')
            seen_open.add(aid)
        if kind not in REQUESTS | TERMINALS:
            continue
        tx = transactions.get(identity)
        source, dest = p['sourceAccountId'], p['destinationAccountId']
        if tx is None or source not in balances or (tx['type'] == 'TRANSFER' and dest not in balances):
            audit.check(False, 'ledger_reference', identity, 'Missing account/transaction for replay')
            continue
        audit.check(source in seen_open and (tx['type'] != 'TRANSFER' or dest in seen_open), 'account_availability', identity, 'Account not opened before transaction')
        if kind in REQUESTS:
            pending[identity] = balances[source]
            continue
        before = balances[source]
        audit.check(p['sourceBalanceBefore'] == before, 'ledger_before', identity, 'Source balance_before differs from replay')
        success = kind in {'TransferCompleted', 'TransactionPosted'}
        cash_in = tx['type'] == 'CASH_IN'
        # Snapshot contract for this generator: no asynchronous settlement.
        audit.check(pending.get(identity) == before, 'request_context', identity, 'Request balance differs from immediate settlement balance')
        can_pay = cash_in or before >= p['amount'] + p['fee']
        audit.check(success == can_pay, 'settlement_status', identity, 'Success/failure inconsistent with funds in schema 2')
        after = before
        if success:
            after += p['amount'] if cash_in else -p['amount'] - p['fee']
        if cash_in:
            audit.check(p['fee'] == 0, 'cash_in_fee', identity, 'Cash-in must have zero fee')
        audit.check(p['sourceBalanceAfter'] == after and after >= 0, 'ledger_after', identity, 'Invalid source balance_after')
        if tx['type'] == 'TRANSFER':
            dest_before = balances[dest]
            dest_after = dest_before + p['amount'] if success else dest_before
            audit.check(p['destinationBalanceBefore'] == dest_before and p['destinationBalanceAfter'] == dest_after,
                        'ledger_destination', identity, 'Destination snapshot differs from replay')
            delta = (p['sourceBalanceAfter'] + p['destinationBalanceAfter']) - (before + dest_before)
            audit.check(delta == (-p['fee'] if success else 0), 'transfer_conservation', identity, 'Transfer does not conserve balances minus fee')
            balances[dest] = dest_after
            if success:
                fees += p['fee']
        balances[source] = after
        replayed += 1
    audit.check(seen_open == accounts.keys(), 'account_open', 'accounts', 'Missing opening events')
    return {'replayed_transactions': replayed, 'successful_transfer_fees': fees, 'final_balances': balances,
            'scope': 'Transaction ledger only; loan events are excluded from balance mutation.'}


def check_history(tables, events, customers, accounts, audit):
    for identity, customer in customers.items():
        credit = customer.get('credit_profile')
        loc = f'customer:{identity}.credit_profile'
        if not fields(audit, credit, {'score': lambda v: integer(v) and 300 <= v <= 850, 'bad_debt': lambda v: type(v) is bool,
            'updated_at': time_value, 'history': lambda v: isinstance(v, list)}, loc):
            continue
        cutoff = timestamp(credit['updated_at']).date()
        history_dates = []
        for n, point in enumerate(credit['history']):
            spec = {'as_of': lambda v: isinstance(v, str) and date.fromisoformat(v) is not None,
                    'score': lambda v: integer(v) and 300 <= v <= 850,
                    'payment_status': lambda v: v in {'on_time', 'late_30', 'late_60', 'late_90'}}
            if fields(audit, point, spec, f'{loc}.history:{n}'):
                d = date.fromisoformat(point['as_of'])
                history_dates.append(d)
                audit.check(d <= cutoff, 'credit_future', loc, 'Credit history exceeds snapshot availability date')
        audit.check(history_dates == sorted(set(history_dates)), 'credit_order', loc, 'Credit history dates must increase uniquely')
        if credit['history'] and isinstance(credit['history'][-1], dict):
            audit.check(credit['score'] == credit['history'][-1].get('score'), 'credit_score', loc, 'Current credit score differs from last history point')
    applications = {r.get('_id'): r for r in tables['loan_applications'] if text(r.get('_id'))}
    loan_ids = {e['aggregate_id'] for e in events if e['event_type'] == 'LoanDisbursed'}
    for e in events:
        kind, p = e['event_type'], e['payload']
        if kind in {'CustomerProfileCreated', 'CreditProfileRecorded'}:
            audit.check(e['aggregate_id'] in customers, 'reference', e['event_id'], 'Unknown customer aggregate')
            customer = customers.get(e['aggregate_id'], {})
            expected = customer if kind == 'CustomerProfileCreated' else customer.get('credit_profile', {})
            audit.check(p == expected, 'credit_snapshot', e['event_id'], 'Customer/credit event differs from initial snapshot')
            credit = p.get('credit_profile', {}) if kind == 'CustomerProfileCreated' else p
            try:
                audit.check(timestamp(credit['updated_at']) <= timestamp(e['occurred_at']), 'credit_future', e['event_id'], 'Credit snapshot available after event')
            except (KeyError, TypeError, ValueError):
                audit.check(False, 'credit_time', e['event_id'], 'Invalid credit availability timestamp')
        if kind == 'LoanApplicationSubmitted':
            audit.check(p['customer_id'] in customers, 'reference', e['event_id'], 'Unknown loan customer')
            audit.warning('loan_submission_outcome', e['event_id'], 'Submitted payload already contains final approval/rejection; excluded from model features.')
        if kind in {'LoanApplicationSubmitted', 'LoanDecisionRecorded'}:
            audit.check(e['aggregate_id'] in applications, 'reference', e['event_id'], 'Unknown loan application aggregate')
            if kind == 'LoanDecisionRecorded':
                audit.check((p['status'] == 'approved' and p['loan_id'] in loan_ids) or (p['status'] == 'rejected' and p['loan_id'] is None),
                            'loan_decision', e['event_id'], 'Decision and disbursement disagree')
        if kind == 'LoanDisbursed':
            audit.check(p['disbursement_account_id'] in accounts and p['customer_id'] in customers,
                        'reference', e['event_id'], 'Unknown loan account/customer')
            audit.check(p['application_id'] in applications, 'reference', e['event_id'], 'Disbursement references unknown application')
            audit.check(p['loan_id'] == e['aggregate_id'], 'loan_id', e['event_id'], 'Loan ID mismatch')
        if kind == 'LoanPaymentPosted':
            audit.check(p['loan_id'] == e['aggregate_id'] and p['loan_id'] in loan_ids, 'loan_id', e['event_id'], 'Loan payment has no disbursement')
    if tables['loans']:
        audit.warning('loan_ledger_excluded', 'loans', 'Loan cash flows are not reflected in transaction balances; all loan fields are excluded from initial model features.')
    audit.warning('same_timestamp_settlement', 'events', 'Generator has immediate settlement; OTP resend, expiry and processing delays are not simulated.')
