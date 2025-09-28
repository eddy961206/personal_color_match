"""색상환/톤 맵 미니 뷰 생성"""
from __future__ import annotations

import math
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from color_utils import hex_to_rgb, rgb_to_hsv01
from palette_loader import TonePalette


def _hsv_to_rgb_arr(h, s, v):
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
    return int(r * 255), int(g * 255), int(b * 255)


def generate_colorwheel_image(
    hex_color: str,
    tone: TonePalette,
    nearest_hex: str | None = None,
    avoid_hex: str | None = None,
    size: int = 320,
) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (250, 250, 250, 255))
    draw = ImageDraw.Draw(canvas)
    radius = size // 2 - 6
    cx = cy = size // 2

    y_grid, x_grid = np.ogrid[-radius:radius, -radius:radius]
    dist = np.sqrt(x_grid**2 + y_grid**2)
    angle = (np.arctan2(y_grid, x_grid) + np.pi) / (2 * np.pi)
    sat = np.clip(dist / radius, 0, 1)
    hue = angle % 1.0

    hsv_stack = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
    flat = hsv_stack.reshape(-1, 3)
    import colorsys

    rgb_list = [_hsv_to_rgb_arr(h, s, v) for h, s, v in flat]
    rgb_arr = np.array(rgb_list, dtype=np.uint8).reshape(hue.shape + (3,))
    mask = dist <= radius
    wheel = Image.new("RGBA", (radius * 2, radius * 2))
    wheel_pixels = wheel.load()
    for y in range(radius * 2):
        for x in range(radius * 2):
            if mask[y, x]:
                wheel_pixels[x, y] = (*rgb_arr[y, x], 255)
            else:
                wheel_pixels[x, y] = (0, 0, 0, 0)
    canvas.paste(wheel, (cx - radius, cy - radius), wheel)

    # 허용 영역 하이라이트
    allowed_mask = (
        (dist <= radius)
        & (sat >= 0.35)
        & (sat <= 0.95)
        & (((hue * 360) >= 20) & ((hue * 360) <= 160))
    )
    overlay = Image.new("RGBA", (radius * 2, radius * 2), (0, 0, 0, 0))
    ov_pixels = overlay.load()
    for y in range(radius * 2):
        for x in range(radius * 2):
            if allowed_mask[y, x]:
                ov_pixels[x, y] = (255, 180, 120, 70)
    canvas.paste(overlay, (cx - radius, cy - radius), overlay)

    def _plot_point(hex_value: str, outline: Tuple[int, int, int], label: str):
        h, s, v = rgb_to_hsv01(*hex_to_rgb(hex_value))
        ang = h * 2 * math.pi
        r = s * radius
        px = int(cx + math.cos(ang) * r)
        py = int(cy + math.sin(ang) * r)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(*outline, 200), outline=(0, 0, 0))
        draw.text((px + 10, py - 4), label, fill=(50, 50, 50))

    _plot_point(hex_color, (255, 80, 80), "입력")
    if nearest_hex:
        _plot_point(nearest_hex, (70, 160, 255), "팔레트")
    if avoid_hex:
        _plot_point(avoid_hex, (120, 120, 120), "주의")

    return canvas
