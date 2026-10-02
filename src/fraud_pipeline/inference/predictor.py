"""Bước 10: Dữ liệu mới -> Model -> Prediction / Risk Score."""
import pandas as pd

def predict(model, features: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    prob = model.predict_proba(features)
    return pd.DataFrame({"risk_score": prob,
                         "prediction": ["fraud" if p >= threshold else "normal" for p in prob]})
