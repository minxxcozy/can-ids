from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from ids.features import build_dataset_from_csv

def train_attack(csv_path: str, window_sec: float, out_path: str):
    print(f"[2] Loading train CSV: {csv_path}")
    X, y_raw, meta, col_info = build_dataset_from_csv(csv_path, window_sec)

    # Only attack windows 사용
    X_attack = X[y_raw != "Normal"]
    y_attack = y_raw[y_raw != "Normal"]

    print("[2] Attack-only label dist:")
    print(y_attack.value_counts())

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_attack)

    X_train, X_val, y_train, y_val = train_test_split(
        X_attack, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("[2] Training 4-class RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    print("[2] Validation report:")
    print(classification_report(y_val, pred, target_names=list(le.classes_)))

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        "model": clf,
        "columns": list(X.columns),
        "label_encoder": le,
        "class_names": list(le.classes_),
        "window_sec": window_sec
    }, out_path)

    print(f"[✓] Saved attack model → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--window-sec", type=float, default=0.02)
    p.add_argument("--out", default="models/attack.pkl")
    args = p.parse_args()
    train_attack(args.csv, args.window_sec, args.out)


if __name__ == "__main__":
    main()