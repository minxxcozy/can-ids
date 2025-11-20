# ids/io_utils.py
import pandas as pd
from typing import List, Optional, Tuple
from .config import (
    TIMESTAMP_CANDIDATES,
    ID_CANDIDATES,
    DATA_CANDIDATES,
    LABEL_CANDIDATES,
    SUBLABEL_CANDIDATES,
    BYTE_PREFIXES,
)


def _find_first_match(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    timestamp_col = _find_first_match(cols, TIMESTAMP_CANDIDATES)
    id_col        = _find_first_match(cols, ID_CANDIDATES)
    data_col      = _find_first_match(cols, DATA_CANDIDATES)
    label_col     = _find_first_match(cols, LABEL_CANDIDATES)
    sublabel_col  = _find_first_match(cols, SUBLABEL_CANDIDATES)

    # Data_0, data_1, byte_0 등 prefix 기반 바이트 컬럼
    byte_cols = [
        c for c in cols
        if any(c.lower().startswith(p) for p in BYTE_PREFIXES)
    ]

    info = {
        "timestamp": timestamp_col,
        "id": id_col,
        "data": data_col,
        "label": label_col,
        "sublabel": sublabel_col,
        "byte_cols": byte_cols,
    }

    return info


def load_csv_with_meta(path: str):
    df = pd.read_csv(path)
    col_info = detect_columns(df)
    return df, col_info