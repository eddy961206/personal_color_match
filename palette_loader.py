"""팔레트 로딩/탐색 유틸"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from color_utils import hex_to_lab
from color_metrics.delta_e import delta_e_ciede2000


@dataclass
class PaletteColor:
    name: str
    hex: str
    tags: Tuple[str, ...]


@dataclass
class TonePalette:
    tone_id: str
    display_name: str
    groups: Dict[str, List[PaletteColor]]

    def all_groups(self) -> Dict[str, List[PaletteColor]]:
        return self.groups


class PaletteRepository:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._cache: Dict[str, TonePalette] = {}

    def list_tones(self) -> List[str]:
        return [p.stem for p in self.base_dir.glob("*.json") if p.stem != "palette.schema"]

    def load(self, tone_id: str) -> TonePalette:
        if tone_id in self._cache:
            return self._cache[tone_id]
        path = self.base_dir / f"{tone_id}.json"
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        groups: Dict[str, List[PaletteColor]] = {}
        for group_name, colors in raw["groups"].items():
            entries = []
            for c in colors:
                entries.append(PaletteColor(name=c["name"], hex=c["hex"].upper(), tags=tuple(c.get("tags", []))))
            groups[group_name] = entries
        palette = TonePalette(tone_id=raw["tone_id"], display_name=raw["display_name"], groups=groups)
        self._cache[tone_id] = palette
        return palette

    def nearest_color(self, tone: TonePalette, hex_color: str, include: Tuple[str, ...]) -> Tuple[PaletteColor, float, str]:
        target = hex_to_lab(hex_color)
        best = None
        best_dist = 1e9
        best_group = ""
        for group in include:
            for color in tone.groups.get(group, []):
                dist = delta_e_ciede2000(target, hex_to_lab(color.hex))
                if dist < best_dist:
                    best = color
                    best_dist = dist
                    best_group = group
        if best is None:
            raise ValueError("팔레트에 색상이 없어.")
        return best, float(best_dist), best_group

    def avoid_distance(self, tone: TonePalette, hex_color: str) -> Tuple[PaletteColor | None, float]:
        target = hex_to_lab(hex_color)
        best = None
        best_dist = 1e9
        for color in tone.groups.get("avoid", []):
            dist = delta_e_ciede2000(target, hex_to_lab(color.hex))
            if dist < best_dist:
                best = color
                best_dist = dist
        return best, float(best_dist)
