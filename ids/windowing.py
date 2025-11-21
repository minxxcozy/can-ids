# ids/windowing.py
# 슬라이딩 윈도우 (Replay-Enhanced)

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def _attach_window_metadata(win: pd.DataFrame) -> pd.DataFrame:
    """
    리플레이 탐지 강화용 윈도우 메타데이터 생성:
    - message_count (윈도우 내 메시지 수)
    - id_unique_count (다양한 ID 개수)
    - avg_payload_len (평균 데이터 길이)
    """

    win = win.copy()

    # 메시지 수
    win["_window_msg_count"] = len(win)

    # ID 고유 개수
    if "id" in win.columns:
        win["_window_id_unique"] = win["id"].nunique()
    else:
        win["_window_id_unique"] = 0

    # Payload 평균 길이
    payload_len = []
    if "data" in win.columns:
        for v in win["data"].astype(str).values:
            payload_len.append(len(v) // 2)  # hex length → byte length
    elif any(col.startswith("byte") for col in win.columns):
        byte_cols = [c for c in win.columns if c.startswith("byte")]
        for _, row in win[byte_cols].iterrows():
            payload_len.append(sum=pd.notna(row).sum())
    else:
        payload_len.append(0)

    win["_window_avg_payload_len"] = float(np.mean(payload_len))

    return win


def make_time_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    window_sec: float = 1.0,
    step_sec: Optional[float] = None,
) -> List[pd.DataFrame]:
    """
    Timestamp 기반 time-sliding windows 생성.
    Replay 공격을 대비하기 위해 아래 기능을 추가:
    - 순서 정렬 및 연속 index 부여 (리플레이 패턴 탐지에 유용)
    - 윈도우 메타데이터 추가 (message_count, id_unique_count 등)
    """

    if step_sec is None:
        step_sec = window_sec  # non-overlapping 기본

    # Timestamp 정렬
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)

    # 윈도우 생성 범위 결정
    t0 = df_sorted[timestamp_col].iloc[0]
    t_end = df_sorted[timestamp_col].iloc[-1]

    windows = []
    start = t0

    while start <= t_end:
        end = start + window_sec

        mask = (df_sorted[timestamp_col] >= start) & (df_sorted[timestamp_col] < end)
        win = df_sorted[mask]

        if not win.empty:
            # 윈도우 내 메시지 순서 index 부여
            win = win.reset_index(drop=True)
            win["_seq_index"] = win.index       # LSTM/CNN에 유용

            # Replay-detection 메타데이터 추가
            win = _attach_window_metadata(win)

            windows.append(win)

        start += step_sec

    return windows