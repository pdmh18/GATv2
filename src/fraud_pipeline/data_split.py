"""Bước 4: Train / Validation / Test, ưu tiên chia theo thời gian để tránh data leakage."""
import pandas as pd

def time_split(df: pd.DataFrame, train=0.7, val=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]
