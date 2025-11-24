import pandas as pd
from typing import List, Dict


def normalize_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Timestamp를 0부터 시작하도록 정규화"""
    df = df.copy()
    df[ts_col] = df[ts_col] - df[ts_col].min()
    return df


def make_time_windows(
    df: pd.DataFrame,
    col_info: Dict[str, str],
    window_sec: float
) -> List[Dict]:
    """
    반환 형식 (features.py와 완전 일치):
    [
        {
            "df": <window dataframe>,
            "start_time": float,
            "end_time": float
        }
    ]
    """

    ts_col = col_info["timestamp"]

    df = normalize_timestamp(df, ts_col)

    ts = df[ts_col].values
    max_t = float(ts.max())

    windows = []
    cur = 0.0

    while cur < max_t:
        start = cur
        end = cur + window_sec

        wdf = df[(df[ts_col] >= start) & (df[ts_col] < end)]

        windows.append({
            "df": wdf,
            "start_time": float(start),
            "end_time": float(end),
        })

        cur = end

    return windows