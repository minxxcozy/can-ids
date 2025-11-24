# models/train_multi.py

from __future__ import annotations
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from ids.features import build_dataset_from_csv
from ids.normal_profile import compute_normal_profile


def build_multiclass_dataset(
    csv_path: str,
    window_sec: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    train.csv → 윈도우 단위 feature + 5-class 라벨(Normal/Fuzzing/DoS/Spoofing/Replay)
    """
    X, y_raw, meta, col_info = build_dataset_from_csv(csv_path, window_sec)
    if y_raw is None:
        raise ValueError("train CSV에서 Label 컬럼을 찾지 못했습니다.")

    # 라벨 문자열 정리 (공백 제거)
    y_clean = y_raw.astype(str).str.strip()
    return X, y_clean


def train_multi(
    csv_path: str,
    window_sec: float = 1.0,
    out_path: str = "models/multi_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    print(f"[+] Building 5-class dataset from: {csv_path}")
    X, y_str = build_multiclass_dataset(csv_path, window_sec)
    print(f"[+] Dataset: X={X.shape}, num_windows={len(X)}")
    print("[+] Label distribution (window-level):")
    print(y_str.value_counts())

    # Normal profile 계산
    print("[+] Computing Normal profile for rule-based reinforcement...")
    normal_profile = compute_normal_profile(X, y_str)
    if normal_profile:
        print("[+] Normal profile stats keys:", list(normal_profile.keys()))
    else:
        print("[!] Normal profile is empty (Normal 데이터가 없을 수 있음).")

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)
    print("[+] Classes:", class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("[+] Training RandomForest (5-class supervised)...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("[+] Validation classification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    artifact = {
        "model": clf,
        "feature_names": list(X.columns),
        "label_encoder": le,
        "class_names": class_names,
        "window_sec": window_sec,
        "normal_profile": normal_profile,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"[✓] 5-class model saved to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Train 5-class CAN IDS model")
    p.add_argument("--csv", required=True, help="train CSV 경로")
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--out", type=str, default="models/multi_model.pkl")
    args = p.parse_args()

    train_multi(args.csv, args.window_sec, args.out)


if __name__ == "__main__":
    main()