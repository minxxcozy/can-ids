# ids/features.py

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from collections import Counter
import math

import numpy as np
import pandas as pd

from .io_utils import load_csv_with_meta
from .windowing import make_time_windows


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent


def _payload_entropy(hex_str: str) -> float:
    if not isinstance(hex_str, str):
        return 0.0
    s = hex_str.replace(" ", "")
    if len(s) == 0:
        return 0.0
    try:
        raw = bytes.fromhex(s)
    except ValueError:
        return 0.0
    return _entropy(Counter(raw))


def compute_window_features(window_df: pd.DataFrame) -> Dict[str, Any]:
    """Fixed window → feature vector"""

    if len(window_df) <= 1:
        # 빈 window 또는 메시지 거의 없음 → 기본값
        return {
            "n_msgs": len(window_df),
            "duration": 1e-6,
            "total_msg_rate": 0,
            "unique_ids": 0,
            "id_entropy": 0,
            "top1_id_ratio": 0,
            "mean_delta_t": 0,
            "std_delta_t": 0,
            "payload_len_mean": 0,
            "payload_len_std": 0,
            "payload_entropy_mean": 0,
            "payload_entropy_std": 0,
            "dlc_mean": 0,
            "dlc_std": 0,
        }

    ts = window_df["Timestamp"].values.astype(float)
    ids = window_df["Arbitration_ID"].astype(str).values
    payloads = window_df["Data"].astype(str).values
    dlcs = window_df["DLC"].values.astype(float)

    duration = float(ts[-1] - ts[0]) or 1e-6
    delta_ts = np.diff(ts)

    # 기본 통계
    n_msgs = len(window_df)
    total_msg_rate = n_msgs / duration

    # ID 분석
    id_counter = Counter(ids)
    unique_ids = len(id_counter)
    id_entropy = _entropy(id_counter)
    top1_id_ratio = max(id_counter.values()) / n_msgs

    # Δt 분석
    mean_delta_t = float(np.mean(delta_ts))
    std_delta_t = float(np.std(delta_ts))

    # payload entropy
    entropies = []
    lengths = []
    for p in payloads:
        s = p.replace(" ", "")
        lengths.append(len(s) // 2)
        entropies.append(_payload_entropy(p))

    payload_len_mean = float(np.mean(lengths))
    payload_len_std = float(np.std(lengths))
    payload_entropy_mean = float(np.mean(entropies))
    payload_entropy_std = float(np.std(entropies))

    feats = {
        "n_msgs": n_msgs,
        "duration": duration,
        "total_msg_rate": total_msg_rate,
        "unique_ids": unique_ids,
        "id_entropy": id_entropy,
        "top1_id_ratio": top1_id_ratio,
        "mean_delta_t": mean_delta_t,
        "std_delta_t": std_delta_t,
        "payload_len_mean": payload_len_mean,
        "payload_len_std": payload_len_std,
        "payload_entropy_mean": payload_entropy_mean,
        "payload_entropy_std": payload_entropy_std,
        "dlc_mean": float(np.mean(dlcs)),
        "dlc_std": float(np.std(dlcs)),
    }

    return feats


def build_dataset_from_csv(
    csv_path: str,
    window_sec: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, str]]:
    """
    최종 윈도우 + feature dataset 생성
    """

    df, col_info = load_csv_with_meta(csv_path)

    # window = (window_df, window_label, start_time, end_time)
    windows = make_time_windows(df, window_sec)

    feature_rows = []
    labels = []
    meta_rows = []

    for window_df, window_label, start_t, end_t in windows:
        feats = compute_window_features(window_df)
        feature_rows.append(feats)
        labels.append(window_label)

        meta_rows.append(
            {
                "start_time": start_t,
                "end_time": end_t,
                "n_msgs": len(window_df),
            }
        )

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="window_label")
    meta = pd.DataFrame(meta_rows)

    return X, y, meta, col_info