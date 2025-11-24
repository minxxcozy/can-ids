# ids/io_utils.py

from typing import Dict, Tuple
import pandas as pd

def load_csv_with_meta(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(csv_path)

    required = ["Timestamp", "Arbitration_ID", "DLC", "Data"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["Timestamp"] = df["Timestamp"].astype(float)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    col_info: Dict[str, str] = {
        "timestamp": "Timestamp",
        "can_id": "Arbitration_ID",
        "payload": "Data",
        "dlc": "DLC"
    }

    if "Label" in df.columns:
        col_info["label"] = "Label"

    return df, col_info