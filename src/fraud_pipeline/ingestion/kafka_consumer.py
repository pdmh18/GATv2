"""Bước 1b -> 1c: Đọc Kafka và ghi vào EventStoreDB."""
import json, os
from kafka import KafkaConsumer
from .eventstore_client import append_event

def consume(topics: list[str]):
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
    )
    for msg in consumer:
        append_event(stream=msg.topic, event=msg.value)
