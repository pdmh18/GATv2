"""Kafka request -> history -> GATv2 -> result; commit only after acknowledged output."""
import json

from ..features.causal import REQUESTS, microseconds
from .contracts import normalize_event


def load_inference_config(path):
    from ..config import load_config, project_path
    config = load_config(path)
    for obj, key in ((config, 'bundle'), (config['history'], 'path')):
        if key in obj:
            obj[key] = str(project_path(obj[key]))
    return config


def handle_message(predictor, history, value):
    event = normalize_event(value)
    if event is None or event['event_type'] not in REQUESTS:
        return None
    prefix = history.read_before(microseconds(event['occurred_at']))
    return predictor.predict(value, prefix)


def consume(consumer, producer, predictor, history, *, output_topic, delivery_timeout=30):
    for message in consumer:
        result = handle_message(predictor, history, message.value)
        if result is not None:
            # At-least-once output: BE must upsert by transaction/model/feature version.
            key = ':'.join(result[k] for k in ('transaction_id', 'model_version', 'feature_version'))
            producer.send(output_topic, key=key.encode(), value=result).get(timeout=delivery_timeout)
        consumer.commit()


def run(config):
    from kafka import KafkaConsumer, KafkaProducer
    from .history import from_config
    from .predictor import ReplayPredictor
    settings = config['kafka']
    if settings['input_topic'] == settings['output_topic']:
        raise ValueError('Kafka input/output topics must differ')
    predictor = ReplayPredictor(config['bundle'], device=config.get('device', 'cpu'))
    history = from_config(config['history'])
    consumer = KafkaConsumer(settings['input_topic'], bootstrap_servers=settings['bootstrap_servers'],
                             group_id=settings['group_id'], enable_auto_commit=False,
                             max_poll_records=1, max_poll_interval_ms=settings['max_poll_interval_ms'],
                             auto_offset_reset=settings['auto_offset_reset'],
                             value_deserializer=lambda v: json.loads(v.decode()))
    producer = KafkaProducer(bootstrap_servers=settings['bootstrap_servers'], acks='all',
                             value_serializer=lambda v: json.dumps(v).encode())
    try:
        consume(consumer, producer, predictor, history, output_topic=settings['output_topic'],
                delivery_timeout=settings['delivery_timeout_seconds'])
    finally:
        consumer.close()
        producer.close()
