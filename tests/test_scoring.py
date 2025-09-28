from __future__ import annotations

import colorsys
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from color_metrics.adjustment import evaluate_color
from config import DEFAULT_CONFIG
from palette_loader import PaletteRepository
from color_utils import hex_to_rgb, rgb_to_hex

repo = PaletteRepository(Path("palettes"))
tone = repo.load("spring_bright")


def score(hex_color: str) -> int:
    return evaluate_color(hex_color, tone, repo, DEFAULT_CONFIG).score


@pytest.mark.parametrize(
    "hex_color",
    ["#FF6B5C", "#FFC107", "#5CD85C", "#6DCFF6"],
)
def test_palette_representatives_high_score(hex_color: str):
    assert score(hex_color) >= 85


@pytest.mark.parametrize(
    "hex_color",
    ["#B5B5B5", "#B08BA0", "#444444"],
)
def test_avoid_colors_low_score(hex_color: str):
    assert score(hex_color) <= 40


def adjust_color(hex_color: str, sat_factor: float = 1.0, val_factor: float = 1.0) -> str:
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    s = max(0.0, min(1.0, s * sat_factor))
    v = max(0.0, min(1.0, v * val_factor))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex(int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))


def test_saturation_value_monotonic_increase():
    base = "#F48D80"
    lowered = adjust_color(base, sat_factor=0.7, val_factor=0.8)
    higher_sat = adjust_color(lowered, sat_factor=1.3, val_factor=1.0)
    higher_val = adjust_color(lowered, sat_factor=1.0, val_factor=1.25)

    baseline_score = score(lowered)
    sat_score = score(higher_sat)
    val_score = score(higher_val)

    assert sat_score >= baseline_score
    assert val_score >= baseline_score
