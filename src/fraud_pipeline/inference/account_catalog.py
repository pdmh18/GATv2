"""Per-request account nodes, without changing the frozen training catalog."""

from ..features.causal import microseconds

ACCOUNT_IDENTITY_POLICY = 'distinct-account-nodes-v1'


def request_account_catalog(bundle_nodes, request, history, cutoff):
    """Add observed transfer accounts as distinct, cold-start-capable nodes.

    Inputs must already be validated and history must be strictly before the
    request. Only account identity/type is learned from the current request;
    graph edges and dynamic node features still use the original graph cutoff.
    Runtime created_at_us is availability for this inference, not an assertion
    about the account's actual opening date. Existing bundle metadata is kept.
    """
    nodes = list(bundle_nodes)
    mapping = {node['id']: index for index, node in enumerate(nodes)}
    observed = {}
    for event in [*history, request]:
        if event['event_type'] not in {'TransferCreated', 'TransferCompleted'}:
            continue
        for field in ('sourceAccountId', 'destinationAccountId'):
            identity = event['payload'][field]
            if not isinstance(identity, str) or not identity.strip() or identity == '__unknown__':
                raise ValueError('Transfer account identity must be nonempty and not reserved')
            if identity not in mapping:
                first_seen = microseconds(event['occurred_at'])
                observed[identity] = min(observed.get(identity, first_seen), first_seen)
    for identity in sorted(observed):
        mapping[identity] = len(nodes)
        # Cold-start account type comes from the request/history contract. No
        # future edge, terminal balance, or fitted statistic is introduced.
        nodes.append({'id': identity, 'type': 'account',
                      'created_at_us': min(observed[identity], cutoff)})
    return nodes, mapping
