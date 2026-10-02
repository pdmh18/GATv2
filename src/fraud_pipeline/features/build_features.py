import pandas as pd
from . import transaction, behavioral, temporal

def build(df: pd.DataFrame) -> pd.DataFrame:
    for m in (transaction, behavioral, temporal):
        df = m.add(df)
    return df
