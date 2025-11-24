# ids/io_utils.py

from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd


def load_csv_with_meta(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    CSV 파일을 로드하고, Timestamp / CAN ID / Payload / DLC / Label 컬럼을 자동 탐지한다.
    기본 기대 형식: Timestamp, Arbitration_ID, DLC, Data, Label
    """
    df = pd.read_csv(csv_path)

    # timestamp 후보
    ts_candidates = ["Timestamp", "timestamp", "time", "Time", "ts"]
    ts_col: Optional[str] = None
    for c in ts_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"Timestamp column not found. Tried: {ts_candidates}")

    # CAN ID 후보
    id_candidates = ["Arbitration_ID", "arbitration_id", "CAN_ID", "can_id", "ID", "id"]
    id_col: Optional[str] = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"CAN ID column not found. Tried: {id_candidates}")

    # Payload 후보
    payload_candidates = ["Data", "data", "Payload", "payload"]
    payload_col: Optional[str] = None
    for c in payload_candidates:
        if c in df.columns:
            payload_col = c
            break
    if payload_col is None:
        raise ValueError(f"Payload column not found. Tried: {payload_candidates}")

    # DLC 후보
    dlc_candidates = ["DLC", "dlc", "Length", "length"]
    dlc_col: Optional[str] = None
    for c in dlc_candidates:
        if c in df.columns:
            dlc_col = c
            break

    # Label 후보 (train에는 있고, test에는 없을 수도 있음)
    label_candidates = ["Label", "label", "Class", "class", "y"]
    label_col: Optional[str] = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    # timestamp 정렬
    df[ts_col] = df[ts_col].astype(float)
    df = df.sort_values(ts_col).reset_index(drop=True)

    col_info: Dict[str, str] = {
        "timestamp": ts_col,
        "can_id": id_col,
        "payload": payload_col,
    }
    if dlc_col is not None:
        col_info["dlc"] = dlc_col
    if label_col is not None:
        col_info["label"] = label_col

    return df, col_info