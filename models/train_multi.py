import argparse
import os
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from ids.features import build_dataset_from_csv
from ids.normal_profile import compute_normal_profile


def build_multiclass_dataset(csv_path: str, window_sec: float):
    X, y_raw, meta, col_info = build_dataset_from_csv(csv_path, window_sec)

    y_clean = y_raw.astype(str).str.strip()
    return X, y_clean


def train_multi(csv_path: str, window_sec: float, out_path: str):
    print(f"[+] Building 5-class dataset from: {csv_path}")
    X, y_str = build_multiclass_dataset(csv_path, window_sec)

    print(f"[+] Dataset: X={X.shape}, num_windows={len(X)}")
    print("[+] Label distribution:")
    print(y_str.value_counts())

    print("[+] Computing Normal profile…")
    normal_profile = compute_normal_profile(X, y_str)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[+] Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=class_names))

    artifact = {
        "model": clf,
        "label_encoder": le,
        "class_names": class_names,
        "feature_names": list(X.columns),
        "window_sec": window_sec,
        "normal_profile": normal_profile,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"[✓] Saved model → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--out", type=str, default="models/multi_model.pkl")
    args = p.parse_args()

    train_multi(args.csv, args.window_sec, args.out)


if __name__ == "__main__":
    main()