from __future__ import annotations
import argparse
import joblib
import pandas as pd

from ids.features import build_message_dataset


# Feature Alignment 
def align_features(X: pd.DataFrame, required_cols):
    """
    train에서 사용한 feature column 구조를
    test에서도 정확히 맞추기 위해 정렬하는 함수.
    row 수는 전혀 건드리지 않음.

    - 없는 컬럼 → 0으로 생성
    - 불필요한 컬럼 → 제거
    - 순서 → train 순서와 동일하게 맞춤
    """
    X = X.copy()

    # 1. train 컬럼 중 test에 없는 것은 추가 (0-filled)
    for col in required_cols:
        if col not in X.columns:
            X[col] = 0

    # 2. test에만 존재하는 쓸데없는 컬럼은 제거
    extra_cols = [col for col in X.columns if col not in required_cols]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    # 3. train 과 동일한 컬럼 순서로 재정렬
    X = X[required_cols]

    return X



# Prediction Pipeline
def predict(csv_path: str, template_path: str,
            binary_path: str, attack_path: str,
            out_path: str, threshold: float):

    print(f"[PRED] Loading test CSV: {csv_path}")
    X, _, df_feat = build_message_dataset(csv_path)

    print("[PRED] Loading models...")
    bin_art = joblib.load(binary_path)
    atk_art = joblib.load(attack_path)

    bin_model = bin_art["model"]
    atk_model = atk_art["model"]
    le = atk_art["label_encoder"]


    # IMPORTANT: feature alignment
    print("[PRED] Aligning feature columns...")

    X_bin = align_features(X, bin_art["columns"])
    X_atk = align_features(X, atk_art["columns"])


    # Stage 1: Normal vs Attack
    print("[PRED] Binary classification...")
    bin_pred = bin_model.predict(X_bin)


    # Stage 2: Attack 4-class
    print("[PRED] 4-class classification...")
    atk_proba = atk_model.predict_proba(X_atk)
    atk_idx = atk_model.predict(X_atk)
    atk_label = le.inverse_transform(atk_idx)
    atk_max = atk_proba.max(axis=1)


    # Combine Prediction
    final_label = []
    for i in range(len(X)):
        if bin_pred[i] == "Normal":
            final_label.append("Normal")
        else:
            if atk_max[i] < threshold:
                final_label.append("Normal")
            else:
                final_label.append(atk_label[i])


    # Final submission = template row-by-row + final_label
    df_template = pd.read_csv(template_path, dtype=str)

    # sanity check: 길이 같아야 한다
    assert len(df_template) == len(final_label), \
        f"Length mismatch: template={len(df_template)}, pred={len(final_label)}"

    df_out = df_template.copy()
    df_out["Label"] = final_label
    df_out = df_out[["Timestamp", "Label"]]

    df_out.to_csv(out_path, index=False)
    print(f"[✓] Saved submission → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--binary", default="models/binary.pkl")
    p.add_argument("--attack", default="models/attack.pkl")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--threshold", type=float, default=0.55)

    args = p.parse_args()

    predict(
        csv_path=args.csv,
        template_path=args.template,
        binary_path=args.binary,
        attack_path=args.attack,
        out_path=args.out,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()