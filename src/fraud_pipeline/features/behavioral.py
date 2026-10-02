"""Behavioral features: tần suất, số giao dịch..."""
import pandas as pd

def add(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("src_acc")["amount"]
    df["src_tx_count"] = g.transform("count")
    df["src_amount_mean"] = g.transform("mean")
    return df
