"""Bước 1b: Đẩy event vào Kafka."""
import json, os
from kafka import KafkaProducer

_producer = None

def send_event(topic: str, event: dict):
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
    _producer.send(topic, event)
