# ids/sequence_builder.py
from typing import Tuple, Any
import numpy as np
import pandas as pd
from .io_utils import load_csv_with_meta


def build_dt_sequences_from_csv(
    csv_path: str,
    seq_len: int = 50,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Δt 시퀀스를 만드는 유틸.
    - seq_len: Δt 길이 (한 샘플의 시퀀스 길이)
    - step: 슬라이딩 step (1이면 한 프레임씩 이동)

    return:
      X: (num_samples, seq_len, 1)
      y: (num_samples,)
    """
    df, col_info = load_csv_with_meta(csv_path)

    t_col = col_info["timestamp"]
    label_col = col_info["label"]

    if t_col is None:
        raise ValueError("Timestamp column not detected.")
    if label_col is None:
        raise ValueError("Label column not detected. (시퀀스 supervised 학습 불가)")

    df_sorted = df.sort_values(by=t_col).reset_index(drop=True)
    ts = df_sorted[t_col].values.astype(float)
    labels = df_sorted[label_col].values

    # Δt: 길이 = N-1
    dts = np.diff(ts)
    n_dts = len(dts)

    X_list = []
    y_list = []

    # dts[start : start+seq_len] + labels[start : start+seq_len+1]
    max_start = n_dts - seq_len + 1
    for start in range(0, max_start, step):
        end = start + seq_len
        dt_window = dts[start:end]  # shape = (seq_len,)

        # 라벨: 프레임 기준으로 majority vote
        frame_start = start
        frame_end = start + seq_len + 1  # dt가 seq_len이면 frame은 seq_len+1
        window_labels = labels[frame_start:frame_end]
        values, counts = np.unique(window_labels, return_counts=True)
        major_label = values[np.argmax(counts)]

        X_list.append(dt_window.reshape(seq_len, 1))
        y_list.append(major_label)

    if not X_list:
        # 데이터 너무 짧은 경우
        return np.empty((0, seq_len, 1), dtype=float), np.empty((0,), dtype=object)

    X = np.stack(X_list)  # (num_samples, seq_len, 1)
    y = np.array(y_list)

    return X, y
