import pandas as pd
from fraud_pipeline.data_split import time_split

def test_time_split_no_leakage():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="h")})
    tr, va, te = time_split(df)
    assert tr.timestamp.max() < va.timestamp.min() < te.timestamp.min()
