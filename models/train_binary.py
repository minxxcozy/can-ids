from __future__ import annotations
import argparse
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ids.features import build_message_dataset


def train_binary(csv_path: str, out_path: str):
    print(f"[1] Loading {csv_path}")

    X, y_raw, _ = build_message_dataset(csv_path)

    # 1) Normal vs Attack
    y_binary = y_raw.apply(lambda x: "Normal" if x == "Normal" else "Attack")

    # 2) Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print("[1] Training RandomForest Binary...")
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    print(classification_report(y_val, pred))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        "model": clf,
        "columns": list(X.columns),
    }, out_path)

    print(f"[✓] Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="models/binary.pkl")
    args = p.parse_args()

    train_binary(args.csv, args.out)


if __name__ == "__main__":
    main()