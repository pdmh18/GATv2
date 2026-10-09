"""Dataset schema 2 event contracts; native transfer export uses schema 1."""
from uuid import UUID

from .core import fields, integer, text, time_value, no_leakage

TYPES = {"TRANSFER", "CASH_IN", "CASH_OUT", "PAYMENT", "DEBIT"}
CHANNELS = {"mobile", "web", "desktop"}
REQUESTS = {"TransferCreated", "TransactionRequested"}
TERMINALS = {"TransferCompleted", "TransferCancelled", "TransactionPosted", "TransactionRejected"}
TRANSFER_TYPES = {"TransferCreated", "TransferCompleted", "TransferCancelled"}
EVENT_AGGREGATES = {
    "AccountOpened": "account", "CustomerProfileCreated": "customer", "CreditProfileRecorded": "customer",
    "LoanApplicationSubmitted": "loan_application", "LoanDecisionRecorded": "loan_application",
    "LoanDisbursed": "loan", "LoanPaymentPosted": "loan",
    **{name: "transaction" for name in REQUESTS | TERMINALS},
}
PAYLOADS = {
    "AccountOpened": {"account_id": text, "customer_id": text, "account_no": text, "opening_balance": integer},
    "CustomerProfileCreated": {"_id": text, "customer_code": text, "profile": lambda v: isinstance(v, dict),
        "identity": lambda v: isinstance(v, dict), "credit_profile": lambda v: isinstance(v, dict),
        "kyc_status": text, "status": text, "created_at": time_value, "updated_at": time_value},
    "CreditProfileRecorded": {"score": lambda v: integer(v) and 300 <= v <= 850,
        "bad_debt": lambda v: type(v) is bool, "updated_at": time_value, "history": lambda v: isinstance(v, list)},
    "LoanApplicationSubmitted": {"application_no": text, "customer_id": text, "loan_package": lambda v: isinstance(v, dict),
        "requested_amount": lambda v: integer(v, 1), "status": lambda v: v in {"approved", "rejected"}},
    "LoanDecisionRecorded": {"status": lambda v: v in {"approved", "rejected"}, "loan_id": lambda v: v is None or text(v)},
    "LoanDisbursed": {"loan_id": text, "application_id": text, "customer_id": text, "disbursement_account_id": text,
        "principal": lambda v: integer(v, 1), "outstanding_principal": integer,
        "term_months": lambda v: integer(v, 1), "active": lambda v: type(v) is bool},
    "LoanPaymentPosted": {"loan_id": text, "installment_no": lambda v: integer(v, 1), "status": lambda v: v == "late",
        "principal_amount": lambda v: integer(v, 1), "interest_amount": integer},
}


def validate_event(audit, event, location):
    spec = {"event_id": text, "aggregate_type": text, "aggregate_id": text, "event_type": text,
            "schema_version": lambda v: type(v) is int and v == 2, "occurred_at": time_value,
            "sequence": lambda v: integer(v, 1), "payload": lambda v: isinstance(v, dict),
            "labels": lambda v: v is None}
    valid = fields(audit, event, spec, location, exact=True)
    if not valid:
        return False
    kind = event.get("event_type")
    if not audit.check(kind in EVENT_AGGREGATES, "event_type", location, "Unsupported event type"):
        return False
    valid = audit.check(event.get("aggregate_type") == EVENT_AGGREGATES[kind], "aggregate_type", location,
                        "Event type and aggregate type disagree") and valid
    payload = event.get("payload")
    if kind in REQUESTS | TERMINALS:
        for key in ('event_id', 'aggregate_id'):
            try:
                UUID(event[key])
            except ValueError:
                valid = audit.check(False, 'uuid', location + '.' + key, 'Transaction/event ID must be UUID') and valid
        payload_spec = {"sourceAccountId": text, "destinationAccountId": text, "amount": lambda v: integer(v, 1),
                        "fee": integer, "currency": lambda v: v == "VND", "channel": lambda v: v in CHANNELS,
                        "status": lambda v: v == {"TransferCreated": "awaiting_otp", "TransactionRequested": "requested",
                            "TransferCompleted": "success", "TransferCancelled": "cancelled",
                            "TransactionPosted": "success", "TransactionRejected": "failed"}[kind]}
        if kind not in TRANSFER_TYPES:
            payload_spec["transactionType"] = lambda v: v in TYPES - {"TRANSFER"}
        if kind in TERMINALS:
            payload_spec.update(sourceBalanceBefore=integer, sourceBalanceAfter=integer)
            if kind in TRANSFER_TYPES:
                payload_spec.update(destinationBalanceBefore=integer, destinationBalanceAfter=integer)
        valid = fields(audit, payload, payload_spec, location + ".payload", exact=True) and valid
        valid = audit.check(no_leakage(payload), "leakage", location, "Risk/label field in detector input") and valid
    else:
        valid = fields(audit, payload, PAYLOADS[kind], location + ".payload", exact=True) and valid
    return valid
