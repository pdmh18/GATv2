"""Explicit adapters for the review input and the native Go banking journal."""
from copy import deepcopy
from uuid import NAMESPACE_OID, UUID, uuid5

from ..features.causal import REQUESTS, microseconds
from ..features.request_context import OPTIONAL_FIELDS
from ..validation.contracts import TERMINALS, TYPES, validate_event
from ..validation.core import Audit, no_leakage

RAW_FIELDS = {'transaction_id', 'source_account', 'destination_account', 'amount',
              'transaction_type', 'channel', 'created_at', *OPTIONAL_FIELDS}
NATIVE_FIELDS = {'eventId', 'transactionId', 'eventType', 'schemaVersion', 'sequence', 'occurredAt', 'payload'}
BANK_NON_TRANSACTION_TYPES = {'BankAccountCreated', 'BankAccountUpdated', 'BankBalanceSeeded',
                              'BankImported', 'LoanDisbursementCredited'}


def unwrap_record(row):
    """Extract only Record.Event; never enrich from the Go terminal/profile snapshot."""
    if not isinstance(row, dict):
        raise ValueError('Input must be an object')
    if 'at' not in row or 'type' not in row:
        return row
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('Invalid banking record identity')
    microseconds(row['at'])
    if row['type'] in BANK_NON_TRANSACTION_TYPES:
        return None
    event = row.get('event')
    if (not isinstance(event, dict) or row['id'] != event.get('eventId')
            or row['type'] != event.get('eventType')
            or microseconds(row['at']) != microseconds(event.get('occurredAt'))):
        raise ValueError('Banking record and nested event metadata disagree')
    return event


def normalize_event(row):
    row = unwrap_record(row)
    if row is None:
        return None
    if not no_leakage({k: v for k, v in row.items() if k != 'labels'}) or row.get('labels') is not None:
        raise ValueError('Labels/risk fields are forbidden in inference input')
    if 'eventType' in row:
        if set(row) != NATIVE_FIELDS or type(row['schemaVersion']) is not int or row['schemaVersion'] != 1:
            raise ValueError('Unsupported native banking event envelope')
        if row['eventType'] in {'TransferOTPUpdated', 'TransferCancelled'}:
            # These native transitions do not change State/graph. Go cancellation
            # has no balance snapshots; do not invent missing terminal balances.
            return None
        if row['eventType'] not in {'TransferCreated', 'TransferCompleted', 'TransferCancelled'}:
            raise ValueError('Unsupported native banking event type')
        event = {'event_id': row['eventId'], 'aggregate_id': row['transactionId'],
                 'aggregate_type': 'transaction', 'schema_version': 2, 'event_type': row['eventType'],
                 'sequence': row['sequence'], 'occurred_at': row['occurredAt'],
                 'payload': deepcopy(row['payload']), 'labels': None}
    elif 'transaction_id' in row:
        required = RAW_FIELDS - set(OPTIONAL_FIELDS)
        if not required <= set(row) or set(row) - RAW_FIELDS:
            raise ValueError('Raw transaction fields must match the review contract; posted_at is not request-time input')
        if row['transaction_type'] not in TYPES or not isinstance(row['channel'], str):
            raise ValueError('Unsupported transaction_type/channel')
        if not isinstance(row['transaction_id'], str) or not row['transaction_id'].strip():
            raise ValueError('transaction_id must be a nonempty string')
        try:
            UUID(row['transaction_id'])
            transaction_id = row['transaction_id']
        except ValueError:
            transaction_id = str(uuid5(NAMESPACE_OID, 'inference:transaction:'+row['transaction_id']))
        kind = 'TransferCreated' if row['transaction_type'] == 'TRANSFER' else 'TransactionRequested'
        payload = {'sourceAccountId': row['source_account'], 'destinationAccountId': row['destination_account'],
                   'amount': row['amount'], 'fee': 0, 'currency': 'VND', 'channel': row['channel'].lower(),
                   'status': 'awaiting_otp' if kind == 'TransferCreated' else 'requested',
                   **{k: row[k] for k in OPTIONAL_FIELDS if k in row}}
        if kind == 'TransactionRequested':
            payload['transactionType'] = row['transaction_type']
        event = {'event_id': str(uuid5(NAMESPACE_OID, f"{transaction_id}:{kind}")),
                 'aggregate_id': transaction_id, 'aggregate_type': 'transaction',
                 'event_type': kind, 'schema_version': 2, 'sequence': 1,
                 'occurred_at': row['created_at'], 'payload': payload, 'labels': None}
    else:
        event = deepcopy(row)
    audit = Audit()
    if not validate_event(audit, event, 'inference'):
        raise ValueError(f'Invalid inference event: {audit.issues}')
    return event if event['event_type'] in REQUESTS | TERMINALS else None
