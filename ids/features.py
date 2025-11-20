# ids/features.py
# 윈도우 하나 -> feature dict 하나로 만드는 함수

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .payload_utils import row_to_bytes, compute_entropy_from_bytes


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

    return feats


def label_for_window(w, col_info):
    # 컬럼 이름
    label_col = col_info["label"]

    # 해당 window 안에서의 unique 라벨 수집
    labels = list(w[label_col].astype(str).unique())

    # Attack 우선 규칙
    if "Attack" in labels:
        return "Attack"

    # Normal만 있는 경우
    if "Normal" in labels:
        return "Normal"

    # 기타 레이블 처리 (예: Fuzzing, Spoofing)
    if len(labels) > 0:
        # Attack이 없는데 다른 공격 서브클래스만 있을 때
        return labels[0]

    return None