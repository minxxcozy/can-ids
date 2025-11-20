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


def label_for_window(
    window: pd.DataFrame,
    col_info: Dict[str, Any],
) -> Optional[str]:
    label_col = col_info["label"]
    if label_col is None:
        return None
    # 윈도우 내 다수결
    values, counts = np.unique(window[label_col].values, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]
