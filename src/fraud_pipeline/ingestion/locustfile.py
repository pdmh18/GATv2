"""Bước 1a: Locust tạo tải. Chạy: locust -f src/fraud_pipeline/ingestion/locustfile.py"""
from locust import User, task, between
from .kafka_producer import send_event
from .simulator import make_transaction

ACCOUNTS = [f"A{i:04d}" for i in range(1000)]

class TxUser(User):
    wait_time = between(0.1, 1)

    @task
    def transact(self):
        send_event("transaction", make_transaction(ACCOUNTS))
