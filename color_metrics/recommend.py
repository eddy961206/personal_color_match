"""대체 색상 추천"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from color_utils import hex_to_rgb, rgb_to_hsv01
from palette_loader import TonePalette


@dataclass
class Recommendation:
    name: str
    hex: str
    reason: str


def _reason_phrase(src_hsv, candidate_hsv) -> str:
    src_h, src_s, src_v = src_hsv
    cand_h, cand_s, cand_v = candidate_hsv
    phrases: List[str] = []
    if cand_s - src_s > 0.08:
        phrases.append("채도↑")
    elif src_s - cand_s > 0.08:
        phrases.append("채도↓")
    if cand_v - src_v > 0.08:
        phrases.append("명도↑")
    elif src_v - cand_v > 0.08:
        phrases.append("명도↓")
    diff = abs((cand_h - src_h + 0.5) % 1.0 - 0.5)
    warm_center = 60 / 360
    src_diff = abs((src_h - warm_center + 0.5) % 1.0 - 0.5)
    cand_diff = abs((cand_h - warm_center + 0.5) % 1.0 - 0.5)
    if cand_diff + 1e-5 < src_diff:
        phrases.append("웜 기울기 보정")
    if not phrases:
        phrases.append("톤 균형 유지")
    return ", ".join(phrases)


def suggest_alternatives(
    hex_color: str,
    tone: TonePalette,
    limit: int = 3,
) -> List[Recommendation]:
    src_hsv = rgb_to_hsv01(*hex_to_rgb(hex_color))
    candidates = []
    for group in ("base", "accent"):
        for color in tone.groups.get(group, []):
            if color.hex.upper() == hex_color.upper():
                continue
            cand_hsv = rgb_to_hsv01(*hex_to_rgb(color.hex))
            diff = abs(src_hsv[1] - cand_hsv[1]) + abs(src_hsv[2] - cand_hsv[2])
            candidates.append((diff, color, cand_hsv))
    candidates.sort(key=lambda x: x[0])
    results: List[Recommendation] = []
    for _, color, cand_hsv in candidates[:limit]:
        results.append(Recommendation(name=color.name, hex=color.hex, reason=_reason_phrase(src_hsv, cand_hsv)))
    return results
