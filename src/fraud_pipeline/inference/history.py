"""History adapters. All readers return a complete prefix, with timestamps < target."""
import json
from pathlib import Path

from ..features.causal import microseconds
from ..ingestion.request_loader import jsonl_rows
from .contracts import unwrap_record


def event_time(event):
    return microseconds(event.get('occurred_at', event.get('occurredAt', event.get('created_at'))))


def _prefix(rows, timestamp_us):
    events = []
    for row in rows:
        event = unwrap_record(row)
        if event is not None and event_time(event) < timestamp_us:
            events.append(event)
    return sorted(events, key=event_time)


class JsonlHistory:
    def __init__(self, path):
        self.path = Path(path)

    def read_before(self, timestamp_us):
        # Do not stop at the first future row: storage order need not be event-time order.
        return _prefix((row for _, row in jsonl_rows(self.path)), timestamp_us)


class EventStoreHistory:
    def __init__(self, client, stream):
        self.client, self.stream = client, stream

    def read_before(self, timestamp_us):
        # Read the complete stream; never use only the target aggregate's history.
        rows = (json.loads(record.data) for record in self.client.get_stream(self.stream))
        return _prefix(rows, timestamp_us)


class MongoHistory:
    """Projection documents: {timestamp_us: int, event: canonical/native event}."""
    def __init__(self, collection):
        self.collection = collection

    def read_before(self, timestamp_us):
        rows = self.collection.find({'timestamp_us': {'$lt': timestamp_us}}, {'_id': 0})
        events = []
        for row in rows:
            event = unwrap_record(row['event'])
            if event is None:
                continue
            if row['timestamp_us'] != event_time(event):
                raise ValueError('Mongo history timestamp disagrees with event')
            events.append(event)
        return sorted(events, key=event_time)


def from_config(config):
    kind = config['kind']
    if kind == 'jsonl':
        return JsonlHistory(config['path'])
    if kind == 'eventstore':
        import os
        from esdbclient import EventStoreDBClient
        return EventStoreHistory(EventStoreDBClient(uri=os.environ[config['uri_env']]), config['stream'])
    if kind == 'mongo':
        import os
        from pymongo import MongoClient
        client = MongoClient(os.environ[config['uri_env']])
        return MongoHistory(client[config['database']][config['collection']])
    raise ValueError('History kind must be jsonl, eventstore or mongo')
