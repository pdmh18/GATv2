"""Inference real-time: đọc Kafka -> dự đoán."""
import json, os
from kafka import KafkaConsumer

def run(model, feature_fn, topic="transaction"):
    c = KafkaConsumer(topic, bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP"),
                      value_deserializer=lambda v: json.loads(v.decode()))
    for msg in c:
        print(msg.value, "->", model.predict_proba(feature_fn(msg.value)))
