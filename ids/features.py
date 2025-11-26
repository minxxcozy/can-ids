import pandas as pd
import numpy as np

WINDOW_SIZE = 20


def parse_arbitration_id(x):
    if pd.isna(x):
        return 0
    s = str(x).strip()
    try:
        # hex 또는 숫자
        if s.lower().startswith("0x") or any(c in s.lower() for c in "abcdef"):
            return int(s, 16)
        return int(float(s))
    except:
        return 0


def parse_data_bytes(s, max_len=8):
    if pd.isna(s):
        return [0] * max_len

    parts = str(s).strip().split()
    vals = []
    for p in parts:
        try:
            vals.append(int(p, 16))
        except:
            vals.append(0)

    # 패딩
    if len(vals) < max_len:
        vals += [0] * (max_len - len(vals))
    return vals[:max_len]


def shannon_entropy(arr):
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def build_message_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Timestamp / DLC 만 숫자로 변환
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0)
    df["DLC"] = pd.to_numeric(df["DLC"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ID 변환
    df["Arb_ID_int"] = df["Arbitration_ID"].apply(parse_arbitration_id)

    # Data → 8 bytes
    data_bytes = df["Data"].apply(parse_data_bytes)
    data_bytes_df = pd.DataFrame(
        data_bytes.tolist(),
        columns=[f"data_{i}" for i in range(8)]
    )
    df = pd.concat([df, data_bytes_df], axis=1)

    # Historics
    df["time_delta"] = df["Timestamp"].diff().fillna(0)
    df["time_delta_id"] = df.groupby("Arb_ID_int")["Timestamp"].diff().fillna(0)
    df["id_count"] = df.groupby("Arb_ID_int").cumcount() + 1
    df["msg_index"] = np.arange(len(df)) + 1
    df["id_freq_so_far"] = df["id_count"] / df["msg_index"]

    # Replay / Spoofing helpers
    id_data_last_ts = {}
    same_payload_dt = []
    same_as_prev_flag = []

    prev_id = None
    prev_data = None

    for _, row in df.iterrows():
        key = (row["Arb_ID_int"], row["Data"])
        ts = row["Timestamp"]

        # 동일 payload 마지막 시점
        if key in id_data_last_ts:
            same_payload_dt.append(ts - id_data_last_ts[key])
        else:
            same_payload_dt.append(0.0)
        id_data_last_ts[key] = ts

        # 바로 이전 메시지와 동일 여부
        if prev_id == row["Arb_ID_int"] and prev_data == row["Data"]:
            same_as_prev_flag.append(1)
        else:
            same_as_prev_flag.append(0)

        prev_id = row["Arb_ID_int"]
        prev_data = row["Data"]

    df["time_since_last_same_payload"] = same_payload_dt
    df["is_same_as_prev_id_data"] = same_as_prev_flag

    # Rolling windows
    df["ArbID_code"], _ = pd.factorize(df["Arb_ID_int"])

    df["id_entropy_window"] = df["ArbID_code"].rolling(
        WINDOW_SIZE, min_periods=1
    ).apply(shannon_entropy, raw=True)

    df["dlc_mean_window"] = df["DLC"].rolling(WINDOW_SIZE, min_periods=1).mean()
    df["time_delta_mean_window"] = df["time_delta"].rolling(WINDOW_SIZE, min_periods=1).mean()
    df["time_delta_std_window"] = df["time_delta"].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)

    byte_cols = [f"data_{i}" for i in range(8)]
    df["data_mean"] = df[byte_cols].mean(axis=1)
    df["data_std"] = df[byte_cols].std(axis=1).fillna(0)

    df = df.fillna(0)
    return df


def build_message_dataset(csv_path: str):
    """
    Train/Test 공용 메시지 단위 데이터셋.
    Row drop 절대 없음.
    """

    # 모든 컬럼을 문자열로 읽어 row drop 방지
    df = pd.read_csv(csv_path, dtype=str)

    df_feat = build_message_features(df)

    feature_cols = [
        "Arb_ID_int", "DLC",
        "time_delta", "time_delta_id",
        "id_count", "msg_index", "id_freq_so_far",
        "time_since_last_same_payload", "is_same_as_prev_id_data",
        "id_entropy_window", "dlc_mean_window",
        "time_delta_mean_window", "time_delta_std_window",
        "data_mean", "data_std",
    ] + [f"data_{i}" for i in range(8)]

    X = df_feat[feature_cols]

    # Label 있을 수 있음 (train)
    if "Label" in df.columns:
        y = df["Label"].astype(str)
    else:
        y = None

    return X, y, df_feat