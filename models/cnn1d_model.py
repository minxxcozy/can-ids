# models/cnn1d_model.py
# Δt 시퀀스용 1D-CNN 분류기

import argparse
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ids.sequence_builder import build_dt_sequences_from_csv
from models.lstm_model import encode_labels, DtSequenceDataset  # 재사용


class CNN1DModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        # input: (B, C, T)
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: (B, T, D) -> (B, C, T)
        x = x.transpose(1, 2)  # D=1 → C=1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        logits = self.fc(x)
        return logits


def train_cnn1d(
    csv_path: str,
    window_size: int = 32,
    step_size: int = 16,
    batch_size: int = 64,
    epochs: int = 10,
    hidden_channels: int = 32,
    kernel_size: int = 3,
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
        raise ValueError("No label column detected in CSV. CNN supervised training 불가.")

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
    model = CNN1DModel(
        input_channels=1,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
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

        # validation
        model.eval()
        all_preds, all_trues = [], []
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

    y_true_labels = [idx2label[i] for i in all_trues]
    y_pred_labels = [idx2label[i] for i in all_preds]
    print("=== CNN1D Classification Report ===")
    print(classification_report(y_true_labels, y_pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    args = parser.parse_args()

    train_cnn1d(
        csv_path=args.csv,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
    )
