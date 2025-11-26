# ids/features.py

import math
from typing import Dict, Any, List
from collections import Counter

import numpy as np
import pandas as pd

from .io_utils import load_csv_with_meta
from .windowing import make_time_windows


# 공격 라벨 우선순위 (train에서만 사용)
ATTACK_PRIORITY = {
    "DoS": 4,
    "Fuzzing": 3,
    "Spoofing": 2,
    "Replay": 1,
    "Normal": 0,
}


#  Entropy Helpers
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


#  Feature Extraction
def compute_window_features(window_df: pd.DataFrame) -> Dict[str, Any]:

    # very small window
    if len(window_df) <= 1:
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

    n_msgs = len(window_df)
    total_msg_rate = n_msgs / duration

    id_counter = Counter(ids)
    unique_ids = len(id_counter)
    id_entropy = _entropy(id_counter)
    top1_id_ratio = max(id_counter.values()) / n_msgs

    mean_delta_t = float(np.mean(delta_ts))
    std_delta_t = float(np.std(delta_ts))

    lengths = []
    entropies = []
    for p in payloads:
        s = p.replace(" ", "")
        lengths.append(len(s) // 2)
        entropies.append(_payload_entropy(p))

    feats = {
        "n_msgs": n_msgs,
        "duration": duration,
        "total_msg_rate": total_msg_rate,
        "unique_ids": unique_ids,
        "id_entropy": id_entropy,
        "top1_id_ratio": top1_id_ratio,
        "mean_delta_t": mean_delta_t,
        "std_delta_t": std_delta_t,
        "payload_len_mean": float(np.mean(lengths)),
        "payload_len_std": float(np.std(lengths)),
        "payload_entropy_mean": float(np.mean(entropies)),
        "payload_entropy_std": float(np.std(entropies)),
        "dlc_mean": float(np.mean(dlcs)),
        "dlc_std": float(np.std(dlcs)),
    }

    return feats


#  Label Assignment (Train only)
def assign_window_label(window_df: pd.DataFrame):
    """
    Train 데이터에는 Label 컬럼이 있음.
    Test(Predict) 데이터에는 없음 → None 반환.
    """
    if "Label" not in window_df.columns:
        return None

    labels = window_df["Label"].unique()
    best = "Normal"
    best_score = 0

    for lb in labels:
        if ATTACK_PRIORITY.get(lb, -1) > best_score:
            best = lb
            best_score = ATTACK_PRIORITY[lb]

    return best


#  Dataset Builder (Train + Predict)
def build_dataset_from_csv(csv_path: str, window_sec: float):
    df, col_info = load_csv_with_meta(csv_path)

    # Test 모드 자동 감지
    skip_label = ("Label" not in df.columns)

    windows = make_time_windows(df, col_info, window_sec)

    feature_rows = []
    labels = []
    meta_rows = []

    for w in windows:
        wdf = w["df"]
        start_t = w["start_time"]
        end_t = w["end_time"]

        # Feature extraction
        feats = compute_window_features(wdf)
        feature_rows.append(feats)

        # Label (train only)
        lbl = assign_window_label(wdf)
        if not skip_label and lbl is not None:
            labels.append(lbl)

        meta_rows.append({
            "start_time": start_t,
            "end_time": end_t,
            "n_msgs": len(wdf),
        })

    X = pd.DataFrame(feature_rows)
    meta = pd.DataFrame(meta_rows)

    # Predict/Test 모드
    if skip_label:
        return X, None, meta, col_info

    # Train 모드
    y = pd.Series(labels, name="window_label")
    return X, y, meta, col_info