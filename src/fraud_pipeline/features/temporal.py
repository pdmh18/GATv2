"""Temporal features: theo thời gian, cửa sổ thời gian..."""
import pandas as pd

def add(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["hour"], df["dayofweek"] = ts.dt.hour, ts.dt.dayofweek
    df["secs_since_last"] = ts.groupby(df["src_acc"]).diff().dt.total_seconds().fillna(0)
    return df
