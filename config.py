"""앱 전역 설정과 채점 파라미터"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoringConfig:
    warm_bias: float = 0.45
    saturation_weight: float = 0.9
    value_weight: float = 0.7
    avoid_penalty_strength: float = 1.2
    avoid_sigma: float = 6.5  # 가우시안 σ (DeltaE 거리)
    contrast_pref: float = 0.3
    contrast_bonus_max: float = 3.0


DEFAULT_CONFIG = ScoringConfig()


TONE_DISPLAY_ORDER = [
    ("spring_bright", "봄 브라이트"),
]
