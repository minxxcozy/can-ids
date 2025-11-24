# models/train_binary.py

from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ids.features import build_dataset_from_csv


def build_binary_labels(y_series: pd.Series) -> pd.Series:
    return y_series.apply(lambda x: "Normal" if x == "Normal" else "Attack")


def train_binary(csv_path: str, window_sec: float, out_path: str):
    print(f"[1] Loading CSV: {csv_path}")

    print("[2] Building windows + features...")
    X, y_raw, meta, col_info = build_dataset_from_csv(csv_path, window_sec)
    print(f"[2] Windows created: {len(X)}")

    print("[3] Converting to binary labels...")
    y_binary = build_binary_labels(y_raw)
    print("[3] Label distribution:")
    print(y_binary.value_counts())

    print("[4] Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"[4] Train: {X_train.shape}, Val: {X_val.shape}")

    print("[5] Training RandomForest (binary)...")
    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("[5] Training complete.")

    print("[6] Evaluating model...")
    pred = clf.predict(X_val)
    print(classification_report(y_val, pred))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        "model": clf,
        "columns": list(X.columns),
        "window_sec": window_sec
    }, out_path)

    print(f"[✓] Saved model → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--window-sec", type=float, default=0.02)
    p.add_argument("--out", default="models/binary.pkl")
    args = p.parse_args()

    train_binary(args.csv, args.window_sec, args.out)


if __name__ == "__main__":
    main()
