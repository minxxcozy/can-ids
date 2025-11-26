# models/predict.py

from __future__ import annotations
import argparse
import joblib
import pandas as pd

from ids.features import build_message_dataset


def predict(csv_path: str, template_path: str,
            binary_path: str, attack_path: str,
            out_path: str, threshold: float):

    print(f"[PRED] Loading test CSV: {csv_path}")
    X, _, df_feat = build_message_dataset(csv_path)

    print("[PRED] Loading models...")
    bin_art = joblib.load(binary_path)
    atk_art = joblib.load(attack_path)

    bin_model = bin_art["model"]
    atk_model = atk_art["model"]
    le = atk_art["label_encoder"]

    X_bin = X[bin_art["columns"]]
    X_atk = X[atk_art["columns"]]

    # Stage 1
    print("[PRED] Binary classification...")
    bin_pred = bin_model.predict(X_bin)

    # Stage 2
    print("[PRED] 4-class classification...")
    atk_proba = atk_model.predict_proba(X_atk)
    atk_idx = atk_model.predict(X_atk)
    atk_label = le.inverse_transform(atk_idx)
    atk_max = atk_proba.max(axis=1)

    final_label = []
    for i in range(len(X)):
        if bin_pred[i] == "Normal":
            final_label.append("Normal")
        else:
            if atk_max[i] < threshold:
                final_label.append("Normal")
            else:
                final_label.append(atk_label[i])

    # Load template.csv (Timestamp, Label)
    print("[PRED] Mapping into template...")
    df_template = pd.read_csv(template_path)

    if "Timestamp" not in df_template.columns:
        raise RuntimeError("template.csv must contain 'Timestamp' column.")

    # Timestamp 기반 merge
    df_feat["pred_label"] = final_label

    df_out = df_template.merge(
        df_feat[["Timestamp", "pred_label"]],
        on="Timestamp",
        how="left"
    )

    df_out["Label"] = df_out["pred_label"].fillna("Normal")
    df_out = df_out[["Timestamp", "Label"]]

    df_out.to_csv(out_path, index=False)
    print(f"[✓] Saved submission → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--binary", default="models/binary.pkl")
    p.add_argument("--attack", default="models/attack.pkl")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--threshold", type=float, default=0.55)

    args = p.parse_args()

    predict(
        csv_path=args.csv,
        template_path=args.template,
        binary_path=args.binary,
        attack_path=args.attack,
        out_path=args.out,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()