"""Bước 1d: Pandas đọc event, làm sạch, tạo dataset."""
import pandas as pd
from .eventstore_client import read_stream

def build_dataset(stream: str = "transaction") -> pd.DataFrame:
    df = pd.DataFrame(read_stream(stream))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
