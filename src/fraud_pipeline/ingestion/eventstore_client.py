"""Bước 1c: Event Store (theo stream, dùng cho CQRS)."""
import json, os
from esdbclient import EventStoreDBClient, NewEvent

_client = None

def _get():
    global _client
    if _client is None:
        _client = EventStoreDBClient(uri=os.getenv("EVENTSTORE_URI"))
    return _client

def append_event(stream: str, event: dict):
    _get().append_to_stream(
        stream_name=stream,
        current_version="any",
        events=[NewEvent(type=event.get("event_type", "event"), data=json.dumps(event).encode())],
    )

def read_stream(stream: str):
    return [json.loads(e.data) for e in _get().get_stream(stream)]
