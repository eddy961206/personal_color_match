"""색상 변환 유틸"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import color as skcolor

HEX_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def validate_hex(s: str) -> bool:
    return bool(HEX_RE.match((s or "").strip()))


def hex_to_rgb(s: str) -> Tuple[int, int, int]:
    m = HEX_RE.match((s or "").strip())
    if not m:
        raise ValueError("올바른 HEX 형식이 아니야. 예: #FF6B5C")
    val = m.group(1)
    return int(val[0:2], 16), int(val[2:4], 16), int(val[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB 범위는 0~255야.")
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_hsv01(r: int, g: int, b: int) -> Tuple[float, float, float]:
    arr = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=float)
    hsv = skcolor.rgb2hsv(arr)
    h, s, v = hsv[0, 0]
    return float(h), float(s), float(v)


def rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    arr = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=float)
    lab = skcolor.rgb2lab(arr)
    return lab[0, 0]


def hex_to_lab(s: str) -> np.ndarray:
    r, g, b = hex_to_rgb(s)
    return rgb_to_lab(r, g, b)


def draw_marker(image: Image.Image, x: int, y: int, color: Tuple[int, int, int], radius: int = 8) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color + (255,), width=3)
    return Image.alpha_composite(img, overlay)
