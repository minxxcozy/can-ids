from __future__ import annotations
import argparse
import pandas as pd
import joblib
import numpy as np

from ids.features import build_dataset_from_csv


def predict(csv_path: str, binary_path: str, attack_path: str,
            out_path: str, window_sec: float, threshold: float):

    print(f"[PRED] Loading test CSV: {csv_path}")

    # 1) 원본 메시지 로드 (최종 제출용 기반)
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.sort_values("Timestamp").reset_index(drop=True)
    df_raw["Label"] = "Normal"   # default label

    # 2) window 단위 feature 생성
    X, _, meta, col_info = build_dataset_from_csv(csv_path, window_sec=window_sec)

    print("[PRED] Loading models...")
    bin_art = joblib.load(binary_path)
    atk_art = joblib.load(attack_path)

    bin_model = bin_art["model"]
    atk_model = atk_art["model"]
    atk_le = atk_art["label_encoder"]

    bin_cols = bin_art["columns"]
    atk_cols = atk_art["columns"]

    # 필요 feature subset
    X_bin = X[bin_cols]
    X_atk = X[atk_cols]

    # 3) Stage 1: Binary classifier
    print("[PRED] Running binary classifier...")
    bin_pred = bin_model.predict(X_bin)

    # 4) Stage 2: Attack classifier
    print("[PRED] Running attack classifier...")
    atk_proba = atk_model.predict_proba(X_atk)
    atk_pred_idx = atk_model.predict(X_atk)
    atk_pred_label = atk_le.inverse_transform(atk_pred_idx)
    atk_max_prob = atk_proba.max(axis=1)

    # 5) Window 단위 최종 예측 label 생성
    final_window_pred = []
    for i in range(len(X)):
        if bin_pred[i] == "Normal":
            final_window_pred.append("Normal")
        else:
            if atk_max_prob[i] < threshold:
                final_window_pred.append("Normal")
            else:
                final_window_pred.append(atk_pred_label[i])

    # 6) Window → Raw 메시지 매핑
    print("[PRED] Mapping window predictions to raw rows...")

    for i in range(len(meta)):
        start_t = meta.iloc[i]["start_time"]
        end_t = meta.iloc[i]["end_time"]
        label = final_window_pred[i]

        # 마지막 window는 <= 로 잡아 데이터 유실 방지
        if i == len(meta) - 1:
            mask = (df_raw["Timestamp"] >= start_t) & (df_raw["Timestamp"] <= end_t)
        else:
            mask = (df_raw["Timestamp"] >= start_t) & (df_raw["Timestamp"] < end_t)

        df_raw.loc[mask, "Label"] = label

    # 7) 제출 파일 저장
    df_raw.to_csv(out_path, index=False)
    print(f"[✓] Saved submission CSV → {out_path}")

    print(df_raw.head())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Test CSV path")
    p.add_argument("--binary", default="models/binary.pkl")
    p.add_argument("--attack", default="models/attack.pkl")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--window-sec", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.55)

    args = p.parse_args()

    predict(
        csv_path=args.csv,
        binary_path=args.binary,
        attack_path=args.attack,
        out_path=args.out,
        window_sec=args.window_sec,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()