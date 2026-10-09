"""Audit contextual generator witnesses; profile metadata is never a model feature."""
from collections import defaultdict, deque
import math

from .core import fields, index_rows, integer, text, timestamp
from .contracts import CHANNELS

SCENARIOS = {'normal', 'contextual_amount', 'velocity_burst', 'graph_fan_in', 'rapid_forwarding'}


def validate_scenarios(tables, manifest, transactions, labels, audit):
    version = manifest.get('scenario_profile')
    if version not in {'contextual-v3', 'contextual-v4'}:
        return
    bounds = None
    if version == 'contextual-v4':
        config = manifest.get('pipeline_config', {})
        stages = config.get('stages', []) if isinstance(config, dict) else []
        for stage in stages if isinstance(stages, list) else []:
            if isinstance(stage, dict) and stage.get('name') == 'transactions':
                options = stage.get('options', {})
                bounds = options.get('outlier_multipliers') if isinstance(options, dict) else None
        if not audit.check(isinstance(bounds, list) and len(bounds) == 2
                           and all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
                           and 0 < bounds[0] <= bounds[1], 'scenario_config', 'manifest',
                           'Contextual v4 requires configured positive amount bounds'):
            return
    profiles = index_rows(tables.get('behavior_profiles', []), 'account_id', audit, 'behavior_profiles')
    accounts = {r['_id'] for r in tables['accounts'] if text(r.get('_id'))}
    audit.check(profiles.keys() == accounts, 'profile_join', 'behavior_profiles', 'One generation profile required per account')
    valid = {}
    for aid, profile in profiles.items():
        spec = {'account_id': text, 'monthly_income': lambda v: integer(v, 1), 'typical_amount': lambda v: integer(v, 1),
            'preferred_hours': lambda v: isinstance(v, list) and 0 < len(set(v)) < 24 and all(type(h) is int and 0 <= h <= 23 for h in v),
            'preferred_channel': lambda v: v in CHANNELS, 'payday': lambda v: integer(v, 1) and v <= 28,
            'activity_weight': lambda v: type(v) in (int, float) and math.isfinite(v) and v > 0}
        if fields(audit, profile, spec, aid, exact=True):
            valid[aid] = profile
    recent, incoming = defaultdict(deque), defaultdict(deque)
    for tx in sorted(transactions.values(), key=lambda r: timestamp(r['created_at'])):
        src, dst = tx['source']['account_id'], tx['destination']['account_id']
        at = timestamp(tx['created_at']).timestamp()
        label = labels.get(tx['transaction_id'], {})
        scenario = label.get('scenario_id')
        if not audit.check(isinstance(scenario, str) and scenario in SCENARIOS, 'scenario_type', tx['transaction_id'], 'Unsupported contextual scenario'):
            continue
        while recent[src] and recent[src][0] < at - 120:
            recent[src].popleft()
        for aid in (src, dst):
            while incoming[aid] and incoming[aid][0][0] < at - 600:
                incoming[aid].popleft()
        if scenario == 'contextual_amount' and src in valid:
            p = valid[src]
            if version == 'contextual-v3':
                witness = timestamp(tx['created_at']).hour not in p['preferred_hours'] and tx['channel'] != p['preferred_channel']
            else:
                low, high = bounds
                witness = (tx['channel'] != p['preferred_channel']
                           and round(p['typical_amount'] * low) <= tx['amount'] <= round(p['typical_amount'] * high))
            audit.check(witness, 'scenario_witness', tx['transaction_id'], 'Contextual amount lacks configured amount/channel witness')
        if scenario == 'velocity_burst':
            audit.check(len(recent[src]) >= 2, 'scenario_witness', tx['transaction_id'], 'Burst has fewer than two prior requests in 120 seconds')
        if scenario == 'graph_fan_in':
            audit.check(tx['type'] == 'TRANSFER' and len({s for _, s in incoming[dst]}) >= 3,
                        'scenario_witness', tx['transaction_id'], 'Fan-in has fewer than three prior distinct sources in 600 seconds')
        if scenario == 'rapid_forwarding':
            audit.check(tx['type'] == 'TRANSFER' and len({s for _, s in incoming[src]}) >= 3,
                        'scenario_witness', tx['transaction_id'], 'Forwarding lacks prior incoming transfer attempts')
        recent[src].append(at)
        if tx['type'] == 'TRANSFER':
            incoming[dst].append((at, src))
