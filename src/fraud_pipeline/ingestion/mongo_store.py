"""Bước 1e: MongoDB lưu dataset đã xử lý cho huấn luyện/đánh giá."""
import os
import pandas as pd
from pymongo import MongoClient

def _db():
    return MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "fraud_ml")]

def save_dataset(df: pd.DataFrame, name: str = "dataset"):
    coll = _db()[name]
    coll.delete_many({})
    coll.insert_many(df.to_dict("records"))

def load_dataset(name: str = "dataset") -> pd.DataFrame:
    return pd.DataFrame(list(_db()[name].find({}, {"_id": 0})))
