from __future__ import annotations
import argparse
import pandas as pd
import joblib
import numpy as np

from ids.features import build_dataset_from_csv


def predict(csv_path: str, binary_path: str, attack_path: str,
            out_path: str, window_sec: float, threshold: float):
    print(f"[PRED] Loading test CSV: {csv_path}")
    X, _, meta, col_info = build_dataset_from_csv(csv_path, window_sec=window_sec)

    print("[PRED] Loading models...")
    bin_art = joblib.load(binary_path)
    atk_art = joblib.load(attack_path)

    # Extract models + feature order
    bin_model = bin_art["model"]
    atk_model = atk_art["model"]
    atk_le = atk_art["label_encoder"]

    # 1) Binary 모델 feature 정렬
    bin_cols = bin_art["columns"]
    X = X[bin_cols]

    # 2) Attack 모델 feature 정렬
    atk_cols = atk_art["columns"]
    X = X[atk_cols]

    # Stage 1 — Binary prediction
    bin_pred = bin_model.predict(X)

    final_pred = []

    for i in range(len(X)):
        if bin_pred[i] == "Normal":
            # Binary가 확신하면 Normal
            label = "Normal"

        else:
            # Attack classifier 확률 기반
            atk_row = X.iloc[[i]]
            proba = atk_model.predict_proba(atk_row)[0]
            max_p = float(np.max(proba))

            if max_p < threshold:
                # 공격 모델이 애매하면 Normal로 보정
                label = "Normal"
            else:
                atk_class = atk_model.predict(atk_row)[0]
                label = atk_le.inverse_transform([atk_class])[0]

        final_pred.append(label)

    df_out = pd.DataFrame({
        "start_time": meta["start_time"],
        "end_time": meta["end_time"],
        "pred": final_pred
    })

    df_out.to_csv(out_path, index=False)
    print(f"[✓] Saved prediction → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--binary", default="models/binary.pkl")
    p.add_argument("--attack", default="models/attack.pkl")
    p.add_argument("--out", default="data/test_pred.csv")
    p.add_argument("--window-sec", type=float, default=0.2,
                   help="Window size (seconds) — train과 동일하게")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Attack classifier 확률 < threshold → Normal 재귀정")

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