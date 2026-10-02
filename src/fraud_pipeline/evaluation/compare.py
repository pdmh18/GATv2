"""Bước 8: So sánh model nghiên cứu vs phương pháp truyền thống."""
import pandas as pd

def compare(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T[["precision", "recall", "f1", "auc_roc"]]

def meets_target(metrics: dict, target: dict) -> bool:
    return metrics["f1"] >= target["f1"] and metrics["auc_roc"] >= target["auc_roc"]
