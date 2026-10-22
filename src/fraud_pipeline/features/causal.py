"""Request-time state: no labels, profiles, balances or future terminal results."""
import math
from collections import Counter, defaultdict, deque
from datetime import datetime
from itertools import groupby
from .history import RollingMedian, RollingMean
from .request_context import context_names, context_values, validate_context_options

TYPES = ('TRANSFER', 'CASH_IN', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'unknown')
CHANNELS = ('mobile', 'web', 'desktop', 'unknown')
REQUESTS = {'TransferCreated', 'TransactionRequested'}
COMPLETED = {'TransferCompleted', 'TransactionPosted'}


def microseconds(value):
    at = datetime.fromisoformat(value)
    if at.tzinfo is None:
        raise ValueError('Event timestamp requires timezone')
    return round(at.timestamp() * 1000000)


def names(options):
    return (['log_amount'] + (['log_fee'] if options['include_fee'] else [])
            + ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'weekend']
            + [f'log_src_request_count_{w}s' for w in options['request_windows_seconds']]
            + ['log_src_history_count', 'log_seconds_since_previous', 'has_previous',
               'log_src_type_completed_median_amount', 'log_src_type_completed_count', 'has_type_completed_history',
               'log_amount_to_type_median', 'type_amount_reference_reliable',
               'new_receiver', 'channel_prior_fraction', 'hour_prior_fraction',
               'log_dst_transfer_attempt_distinct_sources', 'log_src_transfer_attempt_distinct_sources',
               'log_src_received_settled_transfer_amount', 'log_src_transfer_attempt_count', 'cold_start']
            + [f'type_{v}' for v in TYPES] + [f'channel_{v}' for v in CHANNELS]
            + (['log_src_type_completed_mean_amount', 'log_src_type_mean_count', 'has_type_mean_history',
                'log_amount_to_type_mean', 'type_mean_reference_reliable'] if options.get('mean_window_seconds') else [])
            + context_names(options))


def validate_options(options):
    validate_context_options(options)
    if options.get('mean_window_seconds') is not None and (type(options['mean_window_seconds']) is not int or options['mean_window_seconds']<=0):
        raise ValueError('mean_window_seconds must be a positive integer')
    windows = options['request_windows_seconds']
    if not windows or len(set(windows)) != len(windows) or any(type(v) is not int or v <= 0 for v in windows):
        raise ValueError('Request windows must be distinct positive integer seconds')
    for key in ('amount_window_seconds', 'amount_min_history', 'graph_window_seconds', 'neighborhood_window_seconds',
                'node_history_window_seconds', 'node_scaler_samples', 'cold_start_min_requests', 'chunk_size'):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if (options['graph_history'] not in ('settled', 'attempts', 'both')
            or type(options['bidirectional_graph']) is not bool or type(options['include_fee']) is not bool):
        raise ValueError('Invalid graph policy')


class State:
    def __init__(self, options, timezone=None):
        validate_options(options)
        self.options = options
        self.timezone = timezone
        self.requests = {w: defaultdict(deque) for w in options['request_windows_seconds']}
        self.completed = defaultdict(lambda: RollingMedian(options['amount_window_seconds']))
        self.means=defaultdict(lambda: RollingMean(options['mean_window_seconds'])) if options.get('mean_window_seconds') else None
        self.devices=defaultdict(set)
        self.total = Counter()
        self.last = {}
        self.receivers = defaultdict(set)
        self.channels, self.hours = defaultdict(Counter), defaultdict(Counter)
        self.incoming, self.outgoing, self.received = defaultdict(deque), defaultdict(deque), defaultdict(deque)

    def _expire(self, at, src, dst):
        for w, accounts in self.requests.items():
            q = accounts[src]
            while q and q[0] < at - w * 1000000:
                q.popleft()
        cutoff = at - self.options['neighborhood_window_seconds'] * 1000000
        for aid in (src, dst):
            for table in (self.incoming, self.outgoing, self.received):
                q = table[aid]
                while q and q[0][0] < cutoff:
                    q.popleft()

    def score(self, event):
        p, at = event['payload'], microseconds(event['occurred_at'])
        src, dst = p['sourceAccountId'], p['destinationAccountId']
        self._expire(at, src, dst)
        moment = datetime.fromisoformat(event['occurred_at'])
        if self.timezone is not None:
            moment = moment.astimezone(self.timezone)
        hour = moment.hour + moment.minute / 60
        kind = 'TRANSFER' if event['event_type'] == 'TransferCreated' else p.get('transactionType', 'unknown')
        n = self.total[src]
        completed, median = self.completed[(src, kind)].at(at)
        reliable = completed >= self.options['amount_min_history']
        channel = p.get('channel', 'unknown')
        values = [math.log1p(p['amount'])] + ([math.log1p(p['fee'])] if self.options['include_fee'] else []) + [
                  math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                  math.sin(2 * math.pi * moment.weekday() / 7), math.cos(2 * math.pi * moment.weekday() / 7),
                  float(moment.weekday() >= 5)]
        values += [math.log1p(len(self.requests[w][src])) for w in self.options['request_windows_seconds']]
        values += [math.log1p(n), math.log1p((at - self.last[src]) / 1000000) if n else 0., float(n > 0),
                   math.log1p(median), math.log1p(completed), float(completed > 0),
                   math.log1p(p['amount'] / median) if reliable and median > 0 else 0., float(reliable),
                   float(dst not in self.receivers[src]),
                   self.channels[src][channel] / n if n else 0., self.hours[src][moment.hour] / n if n else 0.,
                   math.log1p(len({s for _, s in self.incoming[dst]})),
                   math.log1p(len({s for _, s in self.incoming[src]})),
                   math.log1p(sum(amount for _, amount in self.received[src])),
                   math.log1p(len(self.outgoing[src])), float(n < self.options['cold_start_min_requests'])]
        values += [float(v == (kind if kind in TYPES else 'unknown')) for v in TYPES]
        values += [float(v == (channel if channel in CHANNELS else 'unknown')) for v in CHANNELS]
        if self.means is not None:
            mean_count,mean=self.means[(src,kind)].at(at)
            mean_reliable=mean_count>=self.options['amount_min_history']
            values += [math.log1p(mean),math.log1p(mean_count),float(mean_count>0),
                       math.log1p(p['amount']/mean) if mean_reliable and mean>0 else 0.,float(mean_reliable)]
        if self.options.get('include_request_context',False):
            values += context_values(p,self.devices[src],self.options)
        return values

    def observe(self, event):
        p, at, event_type = event['payload'], microseconds(event['occurred_at']), event['event_type']
        src, dst = p['sourceAccountId'], p['destinationAccountId']
        if event_type in REQUESTS:
            if self.options.get('include_request_context',False) and p.get('device_id') is not None:
                self.devices[src].add(p['device_id'])
            for table in self.requests.values():
                table[src].append(at)
            self.total[src] += 1
            self.last[src] = at
            self.receivers[src].add(dst)
            self.channels[src][p['channel']] += 1
            moment = datetime.fromisoformat(event['occurred_at'])
            if self.timezone is not None:
                moment = moment.astimezone(self.timezone)
            self.hours[src][moment.hour] += 1
            if event_type == 'TransferCreated':
                self.incoming[dst].append((at, src))
                self.outgoing[src].append((at, dst))
        elif event_type in COMPLETED:
            kind = 'TRANSFER' if event_type == 'TransferCompleted' else p.get('transactionType', 'unknown')
            self.completed[(src, kind)].add(at, p['amount'])
            if self.means is not None:self.means[(src,kind)].add(at,p['amount'])
            if event_type == 'TransferCompleted':
                self.received[dst].append((at, p['amount']))


def replay(events, state, observe_event=None):
    """Yield (request, feature vector, prior graph prefix); update a timestamp group afterwards.

    Strictly prior timestamps are visible. All requests at one timestamp share
    the same pre-group history, regardless of lifecycle serialization order.
    """
    previous = None
    graph_prefix = 0
    for at, group in groupby(events, key=lambda e: microseconds(e['occurred_at'])):
        if previous is not None and at < previous:
            raise ValueError('Source events are not chronological')
        rows = list(group)
        transactions = [e for e in rows if e['aggregate_type'] == 'transaction']
        for event in transactions:
            if event['event_type'] in REQUESTS:
                yield event, state.score(event), graph_prefix
        for event in transactions:
            state.observe(event)
            if event['event_type'] in REQUESTS | COMPLETED:
                if observe_event is not None:
                    observe_event(event)
                graph_prefix += 1
        previous = at
