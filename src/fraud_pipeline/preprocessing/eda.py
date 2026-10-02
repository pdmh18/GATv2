"""Bước 2: EDA - phân bố, missing, duplicate, outlier, class imbalance."""
import pandas as pd

def summarize(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "class_balance": df["label"].value_counts(normalize=True).to_dict(),
        "describe": df.describe().to_dict(),
    }
