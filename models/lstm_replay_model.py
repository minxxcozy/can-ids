# models/lstm_replay_model.py
# 리플레이 공격 탐지 강화를 위한 멀티-피처 LSTM 모델
# 기존 lstm_model.py (Δt 단일 특성)는 유지하고, 이 파일은 독립적으로 동작

import argparse
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 리플레이 특화 시퀀스 빌더 (기존 Δt 버전과 독립적으로 동작)
from ids.sequence_builder import build_replay_sequences_from_csv


# PyTorch Dataset
class ReplaySequenceDataset(Dataset):
    """
    N개의 시퀀스 데이터를 PyTorch Dataset 형태로 감싸는 클래스
    X: (N, seq_len, 5)  - 멀티 피처 시퀀스 (Δt + entropy + repeat + sameID + hash)
    y: (N,)             - 라벨
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# LSTM 모델 정의
class LSTMReplayModel(nn.Module):
    """
    리플레이 탐지를 위해 input_dim=5짜리 LSTM 모델
    기존 모델(input_dim=1)과 독립적으로 동작
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_dim,           # Replay 특징 5개 입력
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,               # (B, T, D) 형태 입력
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 양방향일 경우 hidden_dim * 2
        out_dim = hidden_dim * (2 if bidirectional else 1)

        # 최종 클래스 분류기
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]       # 마지막 타임스텝의 hidden state 사용
        logits = self.fc(last_hidden)
        return logits


# 라벨 인코딩 (문자 → 숫자)
def encode_labels(y_raw: np.ndarray) -> Tuple[np.ndarray, Dict[int, Any], Dict[Any, int]]:
    unique_labels = sorted(set(y_raw.tolist()))
    label2idx = {lb: i for i, lb in enumerate(unique_labels)}
    idx2label = {i: lb for lb, i in label2idx.items()}
    y_int = np.array([label2idx[lb] for lb in y_raw])
    return y_int, idx2label, label2idx


# 학습 함수
def train_lstm_replay(
    csv_path: str,
    window_size: int = 32,
    step_size: int = 16,
    batch_size: int = 64,
    epochs: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 1,
    bidirectional: bool = False,
    lr: float = 1e-3,
    test_size: float = 0.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    리플레이 강화용 LSTM 학습 함수
    기존 LSTM과의 차이:
    - 입력: 5차원 시퀀스
    - builder: build_replay_sequences_from_csv()
    """

    # 리플레이 특화 시퀀스 생성
    X, y_raw = build_replay_sequences_from_csv(
        csv_path,
        seq_len=window_size,
        step=step_size,
    )

    if y_raw is None or len(y_raw) == 0:
        raise ValueError("CSV에서 라벨 또는 시퀀스를 찾을 수 없습니다.")

    # 라벨 인코딩
    y_int, idx2label, label2idx = encode_labels(y_raw)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int,
        test_size=test_size,
        random_state=42,
        stratify=y_int,
    )

    train_ds = ReplaySequenceDataset(X_train, y_train)
    test_ds = ReplaySequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(idx2label)

    # Replay Feature 5개
    model = LSTMReplayModel(
        input_dim=5,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_ds)


        # 검증 F1 측정
        model.eval()
        preds_all, trues_all = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)

                preds_all.extend(preds.cpu().numpy())
                trues_all.extend(y_batch.cpu().numpy())

        f1 = f1_score(trues_all, preds_all, average="macro")
        print(f"[Epoch {epoch}/{epochs}] Loss={avg_loss:.4f}  |  F1={f1:.4f}")

    # 최종 결과 리포트
    y_true_labels = [idx2label[i] for i in trues_all]
    y_pred_labels = [idx2label[i] for i in preds_all]

    print("\n=== Replay-Enhanced LSTM Classification Report ===\n")
    print(classification_report(y_true_labels, y_pred_labels))


# CLI 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()

    train_lstm_replay(
        csv_path=args.csv,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
    )