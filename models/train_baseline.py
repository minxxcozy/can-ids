# models/train_baseline.py
# RF + XGB 베이스라인 학습 & 평가

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from ids.io_utils import load_csv_with_meta
from ids.windowing import make_time_windows
from ids.features import window_to_feature_vector, label_for_window


def build_dataset_from_csv(
    csv_path: str,
    window_sec: float = 1.0,
) -> (pd.DataFrame, np.ndarray):
    df, col_info = load_csv_with_meta(csv_path)

    if col_info["timestamp"] is None:
        raise ValueError("Timestamp column not detected.")
    if col_info["label"] is None:
        raise ValueError("Label column not detected. Supervised baseline 불가.")

    windows = make_time_windows(
        df,
        timestamp_col=col_info["timestamp"],
        window_sec=window_sec,
        step_sec=window_sec,
    )

    feature_dicts: List[Dict[str, Any]] = []
    labels: List[Any] = []

    for w in windows:
        feats = window_to_feature_vector(w, col_info)
        label = label_for_window(w, col_info)
        if label is None:
            continue
        feature_dicts.append(feats)
        labels.append(label)

    X = pd.DataFrame(feature_dicts)
    y = np.array(labels)

    return X, y


def train_and_eval_rf_xgb(csv_path: str, window_sec: float = 1.0):
    X, y = build_dataset_from_csv(csv_path, window_sec=window_sec)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average="macro")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")

    print("=== RandomForest ===")
    print("F1 macro:", f1_rf)
    print(classification_report(y_test, y_pred_rf))

    print("=== XGBoost ===")
    print("F1 macro:", f1_xgb)
    print(classification_report(y_test, y_pred_xgb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CAN CSV path")
    parser.add_argument("--window-sec", type=float, default=1.0)
    args = parser.parse_args()

    train_and_eval_rf_xgb(args.csv, window_sec=args.window_sec)
