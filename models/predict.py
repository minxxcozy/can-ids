from __future__ import annotations
import argparse
import pandas as pd
import joblib

from ids.features import build_dataset_from_csv
from ids.reinforce import reinforce_prediction


def predict(csv_path: str, binary_path: str, attack_path: str, out_path: str):
    print(f"[PRED] Loading test CSV: {csv_path}")
    X, _, meta, col_info = build_dataset_from_csv(csv_path, window_sec=0.02)

    print("[PRED] Loading models...")
    bin_art = joblib.load(binary_path)
    atk_art = joblib.load(attack_path)

    bin_model = bin_art["model"]
    atk_model = atk_art["model"]
    atk_le = atk_art["label_encoder"]

    # Stage 1 — Binary (Normal / Attack)
    bin_pred = bin_model.predict(X)

    final_pred = []

    for i in range(len(X)):
        if bin_pred[i] == "Normal":
            # rule-based reinforcement 추가 가능
            label = "Normal"
        else:
            atk_class = atk_model.predict([X.iloc[i]])[0]
            atk_label = atk_le.inverse_transform([atk_class])[0]
            label = atk_label

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
    args = p.parse_args()

    predict(args.csv, args.binary, args.attack, args.out)


if __name__ == "__main__":
    main()