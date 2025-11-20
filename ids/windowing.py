# ids/windowing.py
# 슬라이딩 윈도우

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def make_time_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    window_sec: float = 1.0,
    step_sec: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Timestamp 기반 time-sliding windows 생성."""
    if step_sec is None:
        step_sec = window_sec  # non-overlapping 기본

    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    t0 = df_sorted[timestamp_col].iloc[0]
    t_end = df_sorted[timestamp_col].iloc[-1]

    windows = []
    start = t0
    while start <= t_end:
        end = start + window_sec
        mask = (df_sorted[timestamp_col] >= start) & (df_sorted[timestamp_col] < end)
        win = df_sorted[mask]
        if not win.empty:
            windows.append(win)
        start += step_sec
    return windows
