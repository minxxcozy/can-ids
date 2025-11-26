# ids/windowing.py (FINAL — Replay/Spoofing 최적화)

import pandas as pd

def normalize_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Timestamp를 0부터 시작하도록 정규화"""
    df = df.copy()
    df[ts_col] = df[ts_col] - df[ts_col].min()
    return df


def make_time_windows(df, col_info, window_sec, step_sec=None):
    """
    window_sec: 0.02 strongly recommended (micro anomaly detection)
    step_sec  : default = window_sec (NO overlap)
    """

    ts_col = col_info["timestamp"]
    df = normalize_timestamp(df, ts_col)


    # 1. Overlap 제거
    if step_sec is None:
        step_sec = window_sec

    ts = df[ts_col].values
    max_t = ts.max()

    windows = []
    cur = 0.0

    while cur < max_t:
        start = cur
        end = cur + window_sec

        # window slice
        wdf = df[(df[ts_col] >= start) & (df[ts_col] < end)]

        # 최소 메시지 수 체크 (빈 윈도우 제외)
        if len(wdf) > 1:
            windows.append({
                "df": wdf,
                "start_time": start,
                "end_time": end
            })

        cur += step_sec

    return windows