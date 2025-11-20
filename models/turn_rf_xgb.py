# models/tune_rf_xgb.py
# Optuna로 RF / XGB 하이퍼파라미터 튜닝

import argparse

import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from models.train_baseline import build_dataset_from_csv


def objective_rf(trial, X, y):
    # 하이퍼파라미터 샘플링
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="macro")
    return f1


def objective_xgb(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators", 200, 600)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="macro")
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["rf", "xgb"], required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    print("[+] Loading dataset...")
    X, y = build_dataset_from_csv(args.csv, window_sec=1.0)

    if args.model == "rf":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda tr: objective_rf(tr, X, y), n_trials=args.n_trials)
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda tr: objective_xgb(tr, X, y), n_trials=args.n_trials)

    print("Best F1:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
