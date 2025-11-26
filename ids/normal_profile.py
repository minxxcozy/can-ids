# ids/normal_profile.py (FINAL)

from typing import Dict
import numpy as np
import pandas as pd

def compute_normal_profile(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    mask = (y == "Normal")
    Xn = X[mask]
    if len(Xn) == 0:
        return {}

    prof = {}

    def add(col):
        arr = Xn[col].values.astype(float)
        prof[f"{col}_mean"] = float(np.mean(arr))
        prof[f"{col}_std"] = float(np.std(arr))

    important_cols = [
        "msg_rate", "id_entropy", "top1_id_ratio",
        "mean_dt", "std_dt", "time_delta_id_mean", "time_delta_id_std",
        "payload_len_mean", "payload_entropy_mean",
        "payload_delta_mean", "same_payload_ratio"
    ]

    for c in important_cols:
        if c in Xn.columns:
            add(c)

    for i in range(16):
        col = f"nibble_{i}"
        if col in Xn.columns:
            add(col)

    return prof