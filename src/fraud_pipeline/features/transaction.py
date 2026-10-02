"""Transaction features: số tiền, loại giao dịch, merchant..."""
import numpy as np
import pandas as pd

def add(df: pd.DataFrame) -> pd.DataFrame:
    df["log_amount"] = np.log1p(df["amount"])
    return df
