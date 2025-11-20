# models/lstm_model.py
# Δt 시퀀스용 LSTM 분류기

import argparse
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ids.sequence_builder import build_dt_sequences_from_csv


class DtSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, seq_len, 1)
        y: (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
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
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        # 마지막 타임스텝 hidden 사용
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def encode_labels(y_raw: np.ndarray) -> Tuple[np.ndarray, Dict[int, Any], Dict[Any, int]]:
    unique_labels = sorted(set(y_raw.tolist()))
    label2idx: Dict[Any, int] = {lb: i for i, lb in enumerate(unique_labels)}
    idx2label: Dict[int, Any] = {i: lb for lb, i in label2idx.items()}
    y_int = np.array([label2idx[lb] for lb in y_raw])
    return y_int, idx2label, label2idx


def train_lstm(
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
    X, y_raw, meta = build_dt_sequences_from_csv(
        csv_path,
        window_size=window_size,
        step_size=step_size,
    )

    if y_raw is None:
        raise ValueError("No label column detected in CSV. LSTM supervised training 불가.")

    y_int, idx2label, label2idx = encode_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=test_size,
        random_state=42,
        stratify=y_int,
    )

    train_ds = DtSequenceDataset(X_train, y_train)
    test_ds = DtSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(idx2label)
    model = LSTMModel(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # 간단한 validation
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_trues.extend(y_batch.cpu().numpy().tolist())

        f1 = f1_score(all_trues, all_preds, average="macro")
        print(f"[Epoch {epoch}/{epochs}] Loss={avg_loss:.4f}, F1(macro)={f1:.4f}")

    # 최종 리포트
    y_true_labels = [idx2label[i] for i in all_trues]
    y_pred_labels = [idx2label[i] for i in all_preds]
    print("=== LSTM Classification Report ===")
    print(classification_report(y_true_labels, y_pred_labels))


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

    train_lstm(
        csv_path=args.csv,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
    )
