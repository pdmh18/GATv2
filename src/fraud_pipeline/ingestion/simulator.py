"""Bước 1a: Mô phỏng User / Transaction / Login bằng Mimesis."""
import random
from datetime import datetime
from mimesis import Person

person = Person("en")

def make_transaction(accounts: list[str]) -> dict:
    src, dst = random.sample(accounts, 2)
    return {
        "event_type": "transaction",
        "src_acc": src,
        "dst_acc": dst,
        "amount": round(random.lognormvariate(10, 1.2), 2),
        "timestamp": datetime.utcnow().isoformat(),
        "label": 0,  # TODO: sinh nhãn fraud theo kịch bản
    }

def make_user() -> dict:
    return {"event_type": "user-event", "name": person.full_name(), "email": person.email()}
