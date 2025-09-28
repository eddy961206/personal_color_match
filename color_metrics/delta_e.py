"""CIEDE2000 계산 래퍼"""
from __future__ import annotations

import numpy as np
from skimage import color as skcolor


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000 ΔE 값을 반환한다."""
    d = skcolor.deltaE_ciede2000(lab1.reshape(1, 1, 3), lab2.reshape(1, 1, 3))
    return float(d[0, 0])
