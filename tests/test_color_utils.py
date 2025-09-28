"""color_utils 보조 함수 테스트"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from color_utils import normalize_color_input


def test_normalize_color_input_accepts_rgba_string():
    rgba_value = "rgba(152.8671875, 49.98498715154452, 39.55773711622807, 1)"
    assert normalize_color_input(rgba_value) == "#993228"


def test_normalize_color_input_accepts_plain_hex_without_hash():
    assert normalize_color_input("ff6b5c") == "#FF6B5C"


def test_normalize_color_input_rejects_invalid_string():
    assert normalize_color_input("not-a-color") is None
