# ids/normal_profile.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def compute_normal_profile(X: pd.DataFrame, y_str: pd.Series) -> Dict[str, float]:
    """
    Normal 윈도우들만 모아서 baseline 통계를 만든다.
    이 통계는 reinforce 단계에서 Replay / Spoofing 보강에 사용된다.
    """
    mask_normal = y_str.astype(str) == "Normal"
    Xn = X[mask_normal]
    if len(Xn) == 0:
        # Normal 데이터가 없다면 profile 의미 없음
        return {}

    prof: Dict[str, float] = {}

    def add_stats(prefix: str, series: pd.Series):
        arr = series.values.astype(float)
        prof[f"{prefix}_mean"] = float(np.mean(arr))
        prof[f"{prefix}_std"] = float(np.std(arr))

    # 기본적으로 사용할 주요 feature들 통계
    add_stats("id_entropy", Xn["id_entropy"])
    add_stats("payload_entropy_mean", Xn["payload_entropy_mean"])
    add_stats("payload_entropy_std", Xn["payload_entropy_std"])
    add_stats("top1_id_ratio", Xn["top1_id_ratio"])
    add_stats("total_msg_rate", Xn["total_msg_rate"])
    add_stats("mean_delta_t", Xn["mean_delta_t"])
    add_stats("std_delta_t", Xn["std_delta_t"])

    return prof