"""이미지 오버레이/드래그 평균"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw

from color_utils import clamp, rgb_to_hex


@dataclass
class Marker:
    x: int
    y: int
    color: Tuple[int, int, int]
    label: str


def draw_markers(img: Image.Image, markers: Iterable[Marker]) -> Image.Image:
    base = img.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    for marker in markers:
        x, y = int(marker.x), int(marker.y)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(0, 0, 0), width=3)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=marker.color)
        draw.text((x + 12, y - 4), marker.label, fill=(0, 0, 0))
    return base


def average_color(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Tuple[str, Tuple[int, int, int]]:
    image = img.convert("RGB")
    w, h = image.size
    left = int(clamp(min(x0, x1), 0, w - 1))
    right = int(clamp(max(x0, x1), 0, w - 1))
    top = int(clamp(min(y0, y1), 0, h - 1))
    bottom = int(clamp(max(y0, y1), 0, h - 1))
    crop = image.crop((left, top, right + 1, bottom + 1))
    arr = np.asarray(crop, dtype=np.float32)
    rgb = arr.mean(axis=(0, 1))
    r, g, b = [int(round(v)) for v in rgb]
    return rgb_to_hex(r, g, b), (r, g, b)


def draw_selection_box(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    base = img.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 200, 0), width=3)
    return base
