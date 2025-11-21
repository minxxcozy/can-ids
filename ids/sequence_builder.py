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


def build_replay_sequences_from_csv(
    csv_path: str,
    seq_len: int = 50,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replay 공격 탐지 강화용 멀티피처 시퀀스 생성기.
    기존 build_dt_sequences_from_csv() 와 독립적으로 작동함.
    출력:
        X: (num_samples, seq_len, 5)
        y: (num_samples,)
    """
    from .payload_utils import row_to_bytes, compute_entropy_from_bytes
    import hashlib

    df, col_info = load_csv_with_meta(csv_path)

    t_col = col_info["timestamp"]
    id_col = col_info["id"]
    data_col = col_info["data"]
    byte_cols = col_info["byte_cols"]
    label_col = col_info["label"]

    if t_col is None:
        raise ValueError("Timestamp column not detected.")
    if label_col is None:
        raise ValueError("Label column not detected.")

    df = df.sort_values(by=t_col).reset_index(drop=True)

    # ===== 기본 정보 =====
    ts = df[t_col].values.astype(float)
    labels = df[label_col].values
    ids = df[id_col].values

    # Δt
    dts = np.diff(ts)

    # Entropy
    entropy = []
    for _, row in df.iterrows():
        b = row_to_bytes(row, data_col, byte_cols)
        entropy.append(0.0 if b is None else compute_entropy_from_bytes(b))
    entropy = np.array(entropy)

    # Payload repeat flag
    repeat = []
    prev = None
    for _, row in df.iterrows():
        b = row_to_bytes(row, data_col, byte_cols)
        if b is not None and prev is not None and bytes(b) == prev:
            repeat.append(1)
        else:
            repeat.append(0)
        prev = bytes(b) if b is not None else None
    repeat = np.array(repeat)

    # Same ID as previous
    same_id = np.array([1 if i > 0 and ids[i] == ids[i - 1] else 0 for i in range(len(ids))])

    # Payload hash normalized
    hashes = []
    for _, row in df.iterrows():
        b = row_to_bytes(row, data_col, byte_cols)
        if b is None:
            hashes.append(0.0)
        else:
            h = int(hashlib.sha1(bytes(b)).hexdigest(), 16)
            hashes.append((h % 10000) / 10000.0)
    hashes = np.array(hashes)

    # ===== 시퀀스 생성 =====
    feature_dim = 5
    X_list, y_list = [], []

    num_frames = len(df)
    max_start = num_frames - seq_len

    for start in range(0, max_start, step):
        end = start + seq_len

        X_win = np.stack([
            dts[start:end],          # Δt
            entropy[start:end],      # 엔트로피
            repeat[start:end],       # 반복 플래그
            same_id[start:end],      # ID 반복
            hashes[start:end],       # payload hash
        ], axis=1)

        X_list.append(X_win)

        # Majority label
        frame_labels = labels[start:end+1]
        values, counts = np.unique(frame_labels, return_counts=True)
        y_list.append(values[np.argmax(counts)])

    if not X_list:
        return np.empty((0, seq_len, feature_dim)), np.empty((0,))

    return np.stack(X_list), np.array(y_list)
