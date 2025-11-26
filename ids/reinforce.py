# ids/reinforce.py

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd


# def _get(cls_list: List[str], name: str) -> int:
#     try:
#         return cls_list.index(name)
#     except ValueError:
#         return -1


# def apply_reinforcement(
#     base_label: str,
#     proba_row: np.ndarray,
#     class_names: List[str],
#     feat_row: pd.Series,
#     normal_profile: Dict[str, float] | None,
#     min_confidence: float = 0.6,
# ) -> str:
#     """
#     ML이 예측한 base_label을 Normal 프로파일 기반 rule로 보강한다.

#     아이디어:
#     - confidence(최대 확률)가 충분히 높으면 그대로 신뢰
#     - confidence 낮고 base_label이 Normal일 때,
#       Replay / Spoofing 패턴을 heuristic으로 감지해서 덮어씌우기
#     """
#     if normal_profile is None or len(normal_profile) == 0:
#         return base_label  # 프로파일 없으면 손대지 않음

#     max_conf = float(np.max(proba_row))
#     if max_conf >= min_confidence:
#         # 모델이 확신이 높으면 그대로 사용
#         return base_label

#     # 여기부터는 "애매한 Normal" 같은 케이스를 보완하는 용도
#     if base_label != "Normal":
#         return base_label

#     # feature 추출
#     id_entropy = float(feat_row["id_entropy"])
#     payload_entropy_mean = float(feat_row["payload_entropy_mean"])
#     payload_entropy_std = float(feat_row["payload_entropy_std"])
#     top1_id_ratio = float(feat_row["top1_id_ratio"])
#     total_msg_rate = float(feat_row["total_msg_rate"])

#     # Normal baseline
#     ne_mean = normal_profile.get("id_entropy_mean", 0.0)
#     ne_std = normal_profile.get("id_entropy_std", 1e-6)
#     pe_std_mean = normal_profile.get("payload_entropy_std_mean", 0.0)
#     pe_std_std = normal_profile.get("payload_entropy_std_std", 1e-6)
#     tr_mean = normal_profile.get("total_msg_rate_mean", 0.0)
#     tr_std = normal_profile.get("total_msg_rate_std", 1e-6)
#     t1_mean = normal_profile.get("top1_id_ratio_mean", 0.0)
#     t1_std = normal_profile.get("top1_id_ratio_std", 1e-6)

#     # ───────── Replay 후보 로직 ─────────
#     # - overall traffic은 normal과 거의 비슷
#     # - payload entropy의 분산이 비정상적으로 낮음 (복붙에 가까움)
#     replay_score = 0.0
#     if abs(total_msg_rate - tr_mean) <= 1.0 * tr_std:
#         replay_score += 1.0
#     if abs(id_entropy - ne_mean) <= 1.0 * ne_std:
#         replay_score += 1.0
#     if payload_entropy_std < pe_std_mean - 0.5 * pe_std_std:
#         replay_score += 1.0

#     # ───────── Spoofing 후보 로직 ─────────
#     # - 전체 rate, id_entropy는 normal 범위
#     # - 특정 ID가 비정상적으로 점유율 높음 (top1 ratio 증가)
#     # - payload entropy variance는 normal보다 약간 크거나 비슷
#     spoof_score = 0.0
#     if abs(total_msg_rate - tr_mean) <= 1.0 * tr_std:
#         spoof_score += 1.0
#     if abs(id_entropy - ne_mean) <= 1.0 * ne_std:
#         spoof_score += 1.0
#     if top1_id_ratio > t1_mean + 1.0 * t1_std:
#         spoof_score += 1.0

#     # 간단한 의사결정
#     if replay_score >= 2.5 and replay_score > spoof_score:
#         return "Replay"
#     if spoof_score >= 2.5 and spoof_score > replay_score:
#         return "Spoofing"

#     return base_label

def reinforce_prediction(base_label, *args, **kwargs):
    return base_label