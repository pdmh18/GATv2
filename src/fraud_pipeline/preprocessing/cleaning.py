"""Bước 2: Làm sạch, xử lý missing, encode categorical, chuẩn hóa."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    num = df.select_dtypes("number").columns.drop("label", errors="ignore")
    df[num] = df[num].fillna(df[num].median())
    cat = df.select_dtypes("object").columns.difference(["src_acc", "dst_acc"])
    df = pd.get_dummies(df, columns=list(cat))
    if "amount" in df:
        df["amount_scaled"] = StandardScaler().fit_transform(df[["amount"]])
    return df
