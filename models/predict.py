# models/predict.py

from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import joblib

from ids.features import build_dataset_from_csv
from ids.reinforce import apply_reinforcement


def predict_csv(
    csv_path: str,
    model_path: str = "models/multi_model.pkl",
    window_sec: float | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    print(f"[+] Loading model from: {model_path}")
    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_names: List[str] = artifact["feature_names"]
    label_encoder = artifact["label_encoder"]
    class_names: List[str] = artifact["class_names"]
    trained_window_sec: float = float(artifact.get("window_sec", 1.0))
    normal_profile = artifact.get("normal_profile", {})

    if window_sec is None:
        window_sec = trained_window_sec
    else:
        if abs(window_sec - trained_window_sec) > 1e-6:
            print(
                f"[!] 경고: 모델은 window_sec={trained_window_sec}로 학습되었고, "
                f"지금 {window_sec}로 예측합니다. 가능하면 동일 값을 쓰는 게 좋습니다."
            )

    print(f"[+] Building features from test CSV: {csv_path}")
    X_all, y_raw, meta, col_info = build_dataset_from_csv(csv_path, window_sec)

    # feature 정렬
    missing = [c for c in feature_names if c not in X_all.columns]
    if missing:
        raise ValueError(f"모델에 필요한 feature 컬럼이 없습니다: {missing}")
    X = X_all[feature_names]

    print("[+] Running 5-class ML prediction...")
    probs = model.predict_proba(X.values)
    base_idx = np.argmax(probs, axis=1)
    base_labels = label_encoder.inverse_transform(base_idx)

    final_labels: List[str] = []
    for i, (base_label, proba_row) in enumerate(zip(base_labels, probs)):
        feat_row = X.iloc[i]
        reinforced = apply_reinforcement(
            base_label=base_label,
            proba_row=proba_row,
            class_names=class_names,
            feat_row=feat_row,
            normal_profile=normal_profile,
            min_confidence=0.6,
        )
        final_labels.append(reinforced)

    result_df = pd.DataFrame(
        {
            "window_index": np.arange(len(final_labels)),
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
            "n_msgs": meta["n_msgs"],
            "base_label": base_labels,
            "final_label": final_labels,
        }
    )

    print("[+] Base label distribution:")
    print(result_df["base_label"].value_counts())
    print("[+] Final label distribution (after reinforcement):")
    print(result_df["final_label"].value_counts())

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_csv(out_path, index=False)
        print(f"[✓] Prediction saved to: {out_path}")

    return result_df


def main():
    p = argparse.ArgumentParser(description="Predict 5-class labels with ML + reinforcement")
    p.add_argument("--csv", required=True, help="test CSV 경로")
    p.add_argument("--model", default="models/multi_model.pkl")
    p.add_argument("--window-sec", type=float, default=None)
    p.add_argument("--out", default=None, help="결과 저장 CSV 경로 (예: data/test_pred.csv)")
    args = p.parse_args()

    predict_csv(
        csv_path=args.csv,
        model_path=args.model,
        window_sec=args.window_sec,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()