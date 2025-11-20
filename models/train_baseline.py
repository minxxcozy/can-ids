# models/train_baseline.py
# RF + XGB 베이스라인 학습 & 평가

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from ids.io_utils import load_csv_with_meta
from ids.windowing import make_time_windows
from ids.features import window_to_feature_vector, label_for_window


def build_dataset_from_csv(
    csv_path: str,
    window_sec: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = np.array(labels)
    
    if len(y) == 0:
        raise ValueError("No labels generated. Check label_for_window() or window_sec.")

    print("DEBUG windows:", len(windows))
    print("DEBUG labels:", len(labels))
    print("DEBUG example labels:", labels[:10])

    return X, y


def train_and_eval_rf_xgb(csv_path: str, window_sec: float = 1.0):
    X, y = build_dataset_from_csv(csv_path, window_sec=window_sec)

    # 라벨 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y)

    # StratifiedKFold 준비
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    rf_scores = []
    xgb_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold_idx + 1} =====")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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
        rf_scores.append(f1_rf)

        print("RandomForest F1 (fold):", f1_rf)

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
        xgb_scores.append(f1_xgb)

        print("XGBoost F1 (fold):", f1_xgb)

    # 전체 성능 요약
    print("\n========= FINAL CV RESULTS =========")
    print("RandomForest F1 scores:", rf_scores)
    print("RandomForest F1 mean:", np.mean(rf_scores))

    print("\nXGBoost F1 scores:", xgb_scores)
    print("XGBoost F1 mean:", np.mean(xgb_scores))
    

# def train_and_eval_rf_xgb(csv_path: str, window_sec: float = 1.0):
#     X, y = build_dataset_from_csv(csv_path, window_sec=window_sec)

#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1, random_state=42, stratify=y
#     )

#     # RandomForest
#     rf = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=None,
#         n_jobs=-1,
#         random_state=42,
#     )
#     rf.fit(X_train, y_train)
#     y_pred_rf = rf.predict(X_test)
#     f1_rf = f1_score(y_test, y_pred_rf, average="macro")

#     # XGBoost
#     xgb = XGBClassifier(
#         n_estimators=300,
#         max_depth=5,
#         learning_rate=0.1,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         tree_method="hist",
#         objective="multi:softprob",
#         eval_metric="mlogloss",
#         random_state=42,
#     )
#     xgb.fit(X_train, y_train)
#     y_pred_xgb = xgb.predict(X_test)
#     f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")

#     print("=== RandomForest ===")
#     print("F1 macro:", f1_rf)
#     print(classification_report(y_test, y_pred_rf))

#     print("=== XGBoost ===")
#     print("F1 macro:", f1_xgb)
#     print(classification_report(y_test, y_pred_xgb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CAN CSV path")
    parser.add_argument("--window-sec", type=float, default=1.0)
    args = parser.parse_args()

    train_and_eval_rf_xgb(args.csv, window_sec=args.window_sec)