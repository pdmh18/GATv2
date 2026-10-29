"""Training-data recorder for online predictions.

The record keeps the original request and the model response. It is not a
supervised label: retraining still needs a later outcome joined by
transaction_id, and features must be replayed from event history.
"""
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .response import response_key, validate_response

SCHEMA_VERSION = 1


def build_record(request, response, *, recorded_at=None):
    response = validate_response(response)
    if not isinstance(request, dict):
        raise ValueError('Training request must be an object')
    canonical = json.dumps({'request': request, 'response': response}, sort_keys=True,
                           separators=(',', ':'), ensure_ascii=False)
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat(timespec='microseconds')
    return {
        'schema_version': SCHEMA_VERSION,
        'record_id': hashlib.sha256(canonical.encode()).hexdigest(),
        'recorded_at': timestamp,
        'prediction_key': response_key(response),
        'request': deepcopy(request),
        'response': deepcopy(response),
    }


class TrainingDataSink:
    """JSONL plus optional Kafka topic; duplicates are possible on retry."""

    def __init__(self, path, *, topic=None, fsync=True):
        self.path = Path(path)
        self.topic = topic
        self.fsync = fsync

    def record(self, request, response, *, producer=None, timeout=30):
        record = build_record(request, response)
        self._append(record)
        if self.topic:
            if producer is None:
                raise ValueError('A Kafka producer is required for the training topic')
            producer.send(self.topic, key=record['prediction_key'].encode(),
                          value=record).get(timeout=timeout)
        return record

    def _append(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
            if self.fsync:
                os.fsync(stream.fileno())
