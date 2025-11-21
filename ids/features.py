# ids/features.py
# 윈도우 하나 -> feature dict 하나로 만드는 함수

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .payload_utils import row_to_bytes, compute_entropy_from_bytes


# Δt 기반 통계량
def dt_stats(window: pd.DataFrame, timestamp_col: str) -> Dict[str, float]:
    ts = window[timestamp_col].values
    if len(ts) < 2:
        return {
            "dt_mean": 0.0,
            "dt_std": 0.0,
            "dt_min": 0.0,
            "dt_max": 0.0,
        }
    dts = np.diff(ts)
    return {
        "dt_mean": float(dts.mean()),
        "dt_std": float(dts.std()),
        "dt_min": float(dts.min()),
        "dt_max": float(dts.max()),
    }


# ID 빈도 기반 특징
def id_freq_features(window: pd.DataFrame, id_col: str, top_k: int = 10) -> Dict[str, float]:
    ids, counts = np.unique(window[id_col].values, return_counts=True)
    total = counts.sum()
    idx_sorted = np.argsort(-counts)[:top_k]
    feats = {}
    for rank, idx in enumerate(idx_sorted):
        feats[f"id_top{rank}_value"] = ids[idx]
        feats[f"id_top{rank}_ratio"] = float(counts[idx] / total)
    feats["id_unique_count"] = float(len(ids))
    return feats


# Payload entropy 특징
def entropy_features(
    window: pd.DataFrame,
    data_col: Optional[str],
    byte_cols: List[str],
) -> Dict[str, float]:
    entropies = []
    for _, row in window.iterrows():
        b = row_to_bytes(row, data_col, byte_cols)
        if b is not None:
            entropies.append(compute_entropy_from_bytes(b))

    if not entropies:
        return {
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
        }

    arr = np.array(entropies)
    return {
        "entropy_mean": float(arr.mean()),
        "entropy_std": float(arr.std()),
    }


# Replay 공격에 민감한 Feature 추가
def replay_sensitive_features(window: pd.DataFrame, id_col: str, data_col: str, byte_cols):
    # payload bytes 추출
    payloads = []
    for _, row in window.iterrows():
        b = row_to_bytes(row, data_col, byte_cols)
        if b is not None:
            payloads.append(bytes(b))

    # payload가 하나도 없으면 기본값
    if not payloads:
        return {
            "payload_repeat_ratio": 0.0,
            "payload_change_count": 0.0,
            "payload_change_ratio": 0.0,
            "id_data_combo_repeat_ratio": 0.0,
        }


    # Payload 반복 비율
    unique_payloads = set(payloads)
    payload_repeat_ratio = 1 - (len(unique_payloads) / len(payloads))


    # Payload 변화 횟수 / 비율
    change_count = 0
    for i in range(1, len(payloads)):
        if payloads[i] != payloads[i - 1]:
            change_count += 1
    change_ratio = change_count / max(1, len(payloads) - 1)


    # ID + DATA 조합 반복률
    combos = list(zip(window[id_col].values, payloads))
    unique_combos = len(set(combos))
    combo_repeat_ratio = 1 - (unique_combos / len(combos))

    return {
        "payload_repeat_ratio": float(payload_repeat_ratio),
        "payload_change_count": float(change_count),
        "payload_change_ratio": float(change_ratio),
        "id_data_combo_repeat_ratio": float(combo_repeat_ratio),
    }


# 전체 Feature 생성기
def window_to_feature_vector(
    window: pd.DataFrame,
    col_info: Dict[str, Any],
    top_k_ids: int = 10,
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    t_col = col_info["timestamp"]
    id_col = col_info["id"]
    data_col = col_info["data"]
    byte_cols = col_info["byte_cols"]

    # Δt statistics
    feats.update(dt_stats(window, t_col))

    # ID frequency
    if id_col is not None:
        feats.update(id_freq_features(window, id_col, top_k=top_k_ids))

    # Entropy stats
    feats.update(entropy_features(window, data_col, byte_cols))

    # Replay-sensitive feature
    if id_col is not None:
        feats.update(replay_sensitive_features(window, id_col, data_col, byte_cols))

    return feats


# 라벨 결정
def label_for_window(w, col_info):
    label_col = col_info["label"]
    labels = list(w[label_col].astype(str).unique())

    if "Attack" in labels:
        return "Attack"

    if "Normal" in labels:
        return "Normal"

    if len(labels) > 0:
        return labels[0]

    return None