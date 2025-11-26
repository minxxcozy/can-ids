from __future__ import annotations
import argparse
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from ids.features import build_message_dataset


def train_attack(csv_path: str, out_path: str):
    print(f"[2] Loading {csv_path}")

    X, y_raw, _ = build_message_dataset(csv_path)

    # Attack-only 학습
    mask = y_raw != "Normal"
    X_atk = X[mask]
    y_atk = y_raw[mask]

    print("[2] Attack-only label dist:")
    print(y_atk.value_counts())

    le = LabelEncoder()
    y_enc = le.fit_transform(y_atk)

    X_train, X_val, y_train, y_val = train_test_split(
        X_atk, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("[2] Training XGBoost 4-class...")
    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(le.classes_),
        tree_method="hist",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)

    print(classification_report(y_val, pred, target_names=list(le.classes_)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        "model": clf,
        "label_encoder": le,
        "columns": list(X.columns),
    }, out_path)

    print(f"[✓] Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="models/attack.pkl")
    args = p.parse_args()

    train_attack(args.csv, args.out)


if __name__ == "__main__":
    main()