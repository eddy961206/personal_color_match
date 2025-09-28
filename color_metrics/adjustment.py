"""봄 브라이트 채점 로직"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict

from config import DEFAULT_CONFIG, ScoringConfig
from color_utils import clamp, hex_to_rgb, rgb_to_hsv01
from palette_loader import PaletteRepository, TonePalette

logger = logging.getLogger("spring_bright")


@dataclass
class ScoreBreakdown:
    tone_id: str
    score: int
    delta_e: float
    hsv_boost: float
    warm_alignment: float
    avoid_penalty: float
    contrast_bonus: float
    nearest: Dict[str, str]
    avoid: Dict[str, str] | None


def _warm_alignment(h_deg: float) -> float:
    warm_center = 65.0
    diff = min(abs(h_deg - warm_center), 360.0 - abs(h_deg - warm_center))
    sigma = 70.0
    return math.exp(-0.5 * (diff / sigma) ** 2)


def _gaussian_penalty(dist: float, strength: float, sigma: float) -> float:
    if dist <= 0:
        return min(1.0, strength)
    penalty = strength * math.exp(-0.5 * (dist / max(sigma, 1e-5)) ** 2)
    return clamp(penalty, 0.0, 1.0)


def evaluate_color(
    hex_color: str,
    tone: TonePalette,
    repo: PaletteRepository,
    config: ScoringConfig | None = None,
) -> ScoreBreakdown:
    cfg = config or DEFAULT_CONFIG
    r, g, b = hex_to_rgb(hex_color)
    include = ("base", "accent")
    nearest_color, best_delta, best_group = repo.nearest_color(tone, hex_color, include)

    avoid_color, avoid_delta = repo.avoid_distance(tone, hex_color)

    h, s, v = rgb_to_hsv01(r, g, b)
    h_deg = h * 360.0

    delta_component = clamp(1.0 - best_delta / 45.0, 0.0, 1.0)
    sat_component = clamp((s - 0.35) / 0.45, 0.0, 1.0)
    val_component = clamp((v - 0.5) / 0.4, 0.0, 1.0)
    warm_component = _warm_alignment(h_deg)

    hsv_weight = cfg.saturation_weight + cfg.value_weight + cfg.warm_bias
    hsv_score = 0.0
    if hsv_weight > 0:
        hsv_score = (
            cfg.saturation_weight * sat_component
            + cfg.value_weight * val_component
            + cfg.warm_bias * warm_component
        ) / hsv_weight

    raw_score = 0.62 * delta_component + 0.38 * hsv_score

    avoid_penalty = 0.0
    if avoid_color is not None:
        avoid_penalty = _gaussian_penalty(avoid_delta, cfg.avoid_penalty_strength, cfg.avoid_sigma)
        raw_score *= (1.0 - avoid_penalty)

    # 대비 가산점 평가
    contrast_bonus = 0.0
    for group in ("base", "accent"):
        try:
            pal_color, _, _ = repo.nearest_color(tone, hex_color, (group,))
        except ValueError:
            continue
        nr_h, nr_s, nr_v = rgb_to_hsv01(*hex_to_rgb(pal_color.hex))
        value_diff = abs(v - nr_v)
        target = max(0.0, value_diff - cfg.contrast_pref)
        candidate_bonus = clamp(target / (1.0 - cfg.contrast_pref) * cfg.contrast_bonus_max, -cfg.contrast_bonus_max, cfg.contrast_bonus_max)
        contrast_bonus = max(contrast_bonus, candidate_bonus)

    final_score = clamp(raw_score * 100.0 + contrast_bonus, 0.0, 100.0)
    score_int = int(round(final_score))

    hex_upper = hex_color.upper()
    palette_match = any(color.hex.upper() == hex_upper for group in ("base", "accent") for color in tone.groups.get(group, []))
    if palette_match:
        score_int = 100
        final_score = 100.0
        avoid_penalty = 0.0
        avoid_color = None

    logger.info(
        "[정보] ΔE 최솟값=%.2f(%s), HSV보정=%.1f, 회피penalty=-%.1f, 대비보너스=%.1f → 최종 %d점",
        best_delta,
        nearest_color.name,
        hsv_score * 100,
        avoid_penalty * 100,
        contrast_bonus,
        score_int,
    )

    avoid_payload = None
    if avoid_color is not None:
        avoid_payload = {"name": avoid_color.name, "hex": avoid_color.hex, "distance": avoid_delta}

    return ScoreBreakdown(
        tone_id=tone.tone_id,
        score=score_int,
        delta_e=best_delta,
        hsv_boost=hsv_score * 100.0,
        warm_alignment=warm_component,
        avoid_penalty=avoid_penalty * 100.0,
        contrast_bonus=contrast_bonus,
        nearest={"name": nearest_color.name, "hex": nearest_color.hex, "group": best_group},
        avoid=avoid_payload,
    )
