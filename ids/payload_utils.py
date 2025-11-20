# ids/payload_utils.py
# payload 파싱 & 바이트 배열 추출

from typing import List, Optional, Union
import numpy as np
import pandas as pd


def parse_payload_str(value: Union[str, float, int]) -> Optional[List[int]]:
    """문자열 기반 payload를 [int bytes] 리스트로 변환."""
    if isinstance(value, float) or isinstance(value, int):
        value = str(value)

    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    # "00 7F 01 F4" 형태 (공백)
    if " " in s:
        try:
            return [int(x, 16) for x in s.split()]
        except Exception:
            pass

    # "0x00,0x7F,0x01,0xF4" 형태 (콤마)
    if "," in s:
        try:
            s_clean = s.replace("0x", "").replace("0X", "")
            return [int(x, 16) for x in s_clean.split(",")]
        except Exception:
            pass

    # "007F01F4"처럼 붙은 hex
    if len(s) % 2 == 0:
        try:
            return [int(s[i:i+2], 16) for i in range(0, len(s), 2)]
        except Exception:
            pass

    return None


def row_to_bytes(row, data_col: Optional[str], byte_cols: List[str]) -> Optional[List[int]]:
    """한 row에서 payload 바이트 배열을 추출."""
    # 1) byte_cols(Data_0~)가 있으면 그걸 우선 사용
    if byte_cols:
        try:
            vals = [int(row[c]) for c in byte_cols]
            return vals
        except Exception:
            pass

    # 2) data_col이 있으면 문자열 파싱
    if data_col is not None:
        return parse_payload_str(row[data_col])

    return None


def compute_entropy_from_bytes(byte_list: List[int]) -> float:
    """Shannon entropy."""
    if not byte_list:
        return 0.0
    arr = np.array(byte_list, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())
