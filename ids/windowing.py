import pandas as pd
from typing import List, Tuple, Optional

ATTACK_PRIORITY = {
    "DoS": 4,
    "Fuzzing": 3,
    "Spoofing": 2,
    "Replay": 1,
    "Normal": 0,
}

def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    return df


def assign_window_label(window_df: pd.DataFrame) -> str:
    """공격 한 개라도 있으면 우선순위에 따라 해당 공격으로 라벨링"""
    if len(window_df) == 0:
        return "Normal"

    labels = window_df["Label"].unique()
    best = "Normal"
    best_score = 0

    for lb in labels:
        if ATTACK_PRIORITY.get(lb, -1) > best_score:
            best = lb
            best_score = ATTACK_PRIORITY[lb]

    return best


def make_time_windows(
    df: pd.DataFrame, window_sec: float
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Fixed window (no overlap)
    - timestamp를 0부터 시작하도록 normalization
    - 전체 구간을 window_sec 간격으로 끝까지 탐색
    - window_df가 empty라도 skip하지 않음
    """

    df = normalize_timestamp(df)
    min_t = df["Timestamp"].min()
    max_t = df["Timestamp"].max()

    windows = []
    cur = min_t

    while cur < max_t:
        start = cur
        end = cur + window_sec

        window_df = df[(df["Timestamp"] >= start) &
                       (df["Timestamp"] < end)]

        window_label = assign_window_label(window_df)

        windows.append((window_df, window_label))
        cur = end  # move to next window (no overlap)

    return windows