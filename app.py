"""
봄 브라이트(Spring Bright) 퍼스널컬러 매칭 MVP (Gradio)

요구사항:
- 입력: HEX/RGB 직접 입력, 이미지 업로드, 이미지 클릭 스포이드
- 색상 자동 추출: KMeans (옵션: Gemini API로 LLM 추출)
- 점수 산정: DeltaE(CIEDE2000) + HSV 보정
- 출력: 스와치, 점수/라벨, 가까운 팔레트, 상위 3색 표, 피해야 할 색 경고
- 모든 주석/로그 한국어
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans

from llm_providers import GeminiProvider


# -----------------------------
# 유틸: 색 변환/검증
# -----------------------------

HEX_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def triangular(x: float, a: float, b: float, c: float) -> float:
    """삼각 함수형 멤버십: a~b 상승, b 정점, b~c 하강 (범위 밖 0)."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def validate_hex(s: str) -> bool:
    return bool(HEX_RE.match((s or "").strip()))


def hex_to_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    m = HEX_RE.match(s)
    if not m:
        raise ValueError("올바른 HEX 형식이 아닙니다. 예: #FF6B5C")
    val = m.group(1)
    r = int(val[0:2], 16)
    g = int(val[2:4], 16)
    b = int(val[4:6], 16)
    return r, g, b


def rgb_to_hex(r: int, g: int, b: int) -> str:
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB 범위는 0~255 입니다.")
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


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    # skimage는 (N,3) 또는 (M,N,3) 배열을 받는다.
    d = skcolor.deltaE_ciede2000(lab1.reshape(1, 1, 3), lab2.reshape(1, 1, 3))
    return float(d[0, 0])


# -----------------------------
# 팔레트 로딩 및 최근접 계산
# -----------------------------

def load_palette(path: str = "palette.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nearest_palette(hex_color: str, palette: Dict) -> Tuple[str, float, str]:
    """가장 가까운 팔레트 색상 HEX, DeltaE, 분류(base|accent|avoid)를 반환."""
    lab = hex_to_lab(hex_color)
    best_hex, best_d, best_cat = None, 1e9, ""
    for cat in ("base", "accent", "avoid"):
        for h in palette["spring_bright"][cat]:
            d = delta_e_ciede2000(lab, hex_to_lab(h))
            if d < best_d:
                best_hex, best_d, best_cat = h, d, cat
    return best_hex or "#000000", float(best_d), best_cat


def score_color(hex_color: str, palette: Dict) -> Dict:
    """색상 점수 및 라벨 계산.

    반환 dict:
    - score: int(0~100)
    - label: str
    - nearest_hex: str
    - nearest_delta_e: float
    - avoid_near: bool (DeltaE<20)
    - hsv: (h,s,v) 0..1
    """
    r, g, b = hex_to_rgb(hex_color)
    lab = rgb_to_lab(r, g, b)

    # 팔레트와의 최소 거리
    best_hex, best_d, best_cat = nearest_palette(hex_color, palette)

    # HSV 보정
    _, s, v = rgb_to_hsv01(r, g, b)
    s1 = max(0.0, 1.0 - best_d / 60.0)  # 팔레트 근접도
    s2 = clamp((s - 0.45) / 0.55, 0.0, 1.0)  # 채도 보정
    s3 = triangular(v, 0.60, 0.80, 0.95)  # 밝기 선호
    score = round(100.0 * (0.65 * s1 + 0.20 * s2 + 0.15 * s3))

    # 판정 라벨
    if score >= 90:
        label = "최상(완전 봄 브라이트)"
    elif score >= 75:
        label = "좋음(강추)"
    elif score >= 60:
        label = "보통(상황에 따라 가능)"
    else:
        label = "비추천(탁하거나 톤 안맞음)"

    # 피해야 할 색과의 유사도 체크
    avoid_d = 1e9
    for h in palette["spring_bright"]["avoid"]:
        d = delta_e_ciede2000(lab, hex_to_lab(h))
        if d < avoid_d:
            avoid_d = d
    avoid_near = avoid_d < 20.0

    return {
        "score": int(score),
        "label": label,
        "nearest_hex": best_hex,
        "nearest_delta_e": float(best_d),
        "avoid_near": bool(avoid_near),
        "hsv": (float(_), float(s), float(v)),
    }


# -----------------------------
# 이미지 처리: KMeans / 픽셀 픽
# -----------------------------

def _resize_longest_side(pil_img: Image.Image, longest: int = 512) -> Image.Image:
    # 긴 변을 longest로 맞추고 비율 유지
    w, h = pil_img.size
    if max(w, h) <= longest:
        return pil_img
    if w >= h:
        new_w = longest
        new_h = int(h * (longest / w))
    else:
        new_h = longest
        new_w = int(w * (longest / h))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def extract_colors_kmeans(pil_img: Image.Image, k: int = 5) -> List[Tuple[str, int]]:
    """KMeans로 지배적 색 추출.
    반환: List[(HEX, 픽셀수)] (채도/밝기 필터 후, 픽셀수 내림차순)
    """
    # 전처리: 리사이즈, RGB 정규화
    img = _resize_longest_side(pil_img.convert("RGB"), 512)
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    # KMeans (랜덤 초기화 고정)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_  # (k,3) 0..1

    # 각 클러스터 픽셀 수
    counts = np.bincount(labels, minlength=k)

    # HSV로 저채도/극저밝기 필터링
    hsv_centers = skcolor.rgb2hsv(centers.reshape(1, k, 3))[0]
    results: List[Tuple[str, int]] = []
    for i in range(k):
        h_, s_, v_ = hsv_centers[i]
        if s_ < 0.2 or v_ < 0.2:
            # 배경/그늘로 간주하고 제외
            continue
        r, g, b = (centers[i] * 255).round().astype(int)
        hex_c = rgb_to_hex(int(r), int(g), int(b))
        results.append((hex_c, int(counts[i])))

    # 픽셀 수 기준 내림차순 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def pick_color_at(pil_img: Image.Image, x: int, y: int) -> str:
    """이미지 좌표(x,y) 픽셀에서 HEX 색상 추출."""
    img = pil_img.convert("RGB")
    w, h = img.size
    # Gradio select의 좌표는 이미지 픽셀 좌표 기준
    x = int(clamp(x, 0, w - 1))
    y = int(clamp(y, 0, h - 1))
    r, g, b = img.getpixel((x, y))
    return rgb_to_hex(r, g, b)


# -----------------------------
# HTML 렌더링 유틸 (스와치/표)
# -----------------------------

def render_swatch(hex_color: str, subtitle: str = "") -> str:
    return f'''
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="width:120px;height:120px;border:2px solid #ddd;border-radius:8px; background:{hex_color}"></div>
      <div style="font-size:14px;line-height:1.6">
        <div><b>선택 색상</b> {subtitle}</div>
        <div>HEX: <code>{hex_color}</code></div>
      </div>
    </div>
    '''


def render_score(score: int, label: str) -> str:
    return f'''
    <div style="display:flex;align-items:center;gap:12px;">
      <progress value="{score}" max="100" style="width:240px;height:16px"></progress>
      <div style="font-weight:600;">{score} / 100</div>
      <div style="padding:4px 8px;border-radius:9999px;border:1px solid #ddd;background:#f6f6f6;">{label}</div>
    </div>
    '''


def render_nearest(hex_color: str, delta_e: float, cat: str) -> str:
    name = {"base": "베이스", "accent": "엑센트", "avoid": "피해야 할"}.get(cat, cat)
    return f'''
    <div style="display:flex;align-items:center;gap:12px;">
      <div>가장 가까운 팔레트({name}):</div>
      <div style="width:24px;height:24px;border:1px solid #ccc;border-radius:4px;background:{hex_color}"></div>
      <code>{hex_color}</code>
      <div style="color:#666">DeltaE={delta_e:.1f}</div>
    </div>
    '''


def render_top_colors(rows: List[Tuple[str, int, int, float]]) -> str:
    # rows: (HEX, 픽셀수, 점수, deltaE)
    if not rows:
        return ""
    trs = []
    for hex_c, cnt, score, de in rows[:3]:
        trs.append(
            f"""
            <tr>
              <td><div style='width:20px;height:20px;border:1px solid #ccc;border-radius:4px;background:{hex_c}'></div></td>
              <td><code>{hex_c}</code></td>
              <td style='text-align:right'>{score}</td>
              <td style='text-align:right'>{de:.1f}</td>
              <td style='text-align:right'>{cnt}</td>
            </tr>
            """
        )
    table = f"""
    <div>
      <div style='margin-bottom:6px;font-weight:600'>상위 3색 후보</div>
      <table style='border-collapse:collapse;width:100%;'>
        <thead>
          <tr style='text-align:left;border-bottom:1px solid #ddd'>
            <th>색</th><th>HEX</th><th style='text-align:right'>점수</th><th style='text-align:right'>DeltaE</th><th style='text-align:right'>픽셀수</th>
          </tr>
        </thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </div>
    """
    return table


def render_avoid_warning(show: bool) -> str:
    if not show:
        return ""
    return (
        "<div style='display:inline-block;padding:6px 10px;border-radius:8px;"
        "background:#FFF3CD;border:1px solid #FFEEBA;color:#8A6D3B;'>"
        "경고: 피해야 할 색상과 매우 유사합니다 (DeltaE<20)" "</div>"
    )


# -----------------------------
# Gradio 앱 로직
# -----------------------------

PALETTE = load_palette()


def sync_from_hex(hex_in: str):
    """HEX를 기준으로 RGB/스와치를 동기화."""
    try:
        if not validate_hex(hex_in):
            gr.Warning("HEX 형식이 올바르지 않습니다. 예: #FF6B5C")
            return gr.update(), gr.update(), gr.update(), gr.update(value=render_swatch("#FFFFFF", "(올바른 HEX를 입력하세요)"))
        r, g, b = hex_to_rgb(hex_in)
        sw = render_swatch(hex_in, f"(RGB: {r},{g},{b})")
        return int(r), int(g), int(b), gr.update(value=sw)
    except Exception as e:
        gr.Warning(f"입력 오류: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update(value=render_swatch("#FFFFFF", "(입력 오류)"))


def sync_from_rgb(r: float, g: float, b: float):
    """RGB를 기준으로 HEX/스와치를 동기화."""
    try:
        ri, gi, bi = int(r), int(g), int(b)
        if not (0 <= ri <= 255 and 0 <= gi <= 255 and 0 <= bi <= 255):
            gr.Warning("RGB 값은 0~255 범위여야 합니다.")
            return gr.update(), gr.update(value=render_swatch("#FFFFFF", "(RGB 범위 오류)"))
        hx = rgb_to_hex(ri, gi, bi)
        sw = render_swatch(hx, f"(RGB: {ri},{gi},{bi})")
        return hx, gr.update(value=sw)
    except Exception as e:
        gr.Warning(f"입력 오류: {e}")
        return gr.update(), gr.update(value=render_swatch("#FFFFFF", "(입력 오류)"))


def on_image_select(img: Image.Image, evt: gr.SelectData):
    """이미지 클릭 이벤트: 해당 픽셀 색을 HEX/RGB로 반영."""
    try:
        if img is None:
            gr.Warning("이미지가 없습니다. 먼저 업로드하세요.")
            return gr.update(), gr.update(), gr.update(), gr.update()
        x, y = int(evt.index[0]), int(evt.index[1])
        hx = pick_color_at(img, x, y)
        r, g, b = hex_to_rgb(hx)
        sw = render_swatch(hx, f"(RGB: {r},{g},{b})")
        return hx, int(r), int(g), int(b), gr.update(value=sw)
    except Exception as e:
        gr.Warning(f"픽셀 선택 실패: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def analyze(hex_in: str, r: float, g: float, b: float, img: Optional[Image.Image], api_key_in: str):
    """분석하기: 단일 색 또는 이미지에서 상위 3색을 평가하여 결과를 출력."""
    try:
        key_ui = (api_key_in or "").strip()
        key_env = os.environ.get("GEMINI_API_KEY", "").strip()
        api_key = key_ui or key_env  # UI 우선, 없으면 ENV

        chosen_hex: Optional[str] = None
        hex_valid = validate_hex(hex_in)
        rgb_valid = (r is not None and g is not None and b is not None and 0 <= int(r) <= 255 and 0 <= int(g) <= 255 and 0 <= int(b) <= 255)

        # 우선순위: HEX 유효 -> RGB 유효 -> 이미지 추출
        if hex_valid:
            chosen_hex = hex_in.strip().upper()
        elif rgb_valid:
            chosen_hex = rgb_to_hex(int(r), int(g), int(b))
        elif img is not None:
            # LLM 또는 KMeans로 상위 색 추출
            hex_list: List[str] = []
            used_llm = False
            krows: List[Tuple[str, int]] = []
            if api_key:
                try:
                    provider = GeminiProvider(api_key=api_key)
                    hex_list = provider.get_main_colors(img)
                    used_llm = True
                except Exception as e:
                    print("[경고] LLM 추출 실패, KMeans로 폴백합니다:", str(e))
            if not hex_list:
                # KMeans 폴백
                krows = extract_colors_kmeans(img, k=5)
                hex_list = [h for h, _ in krows[:3]]

            # 상위색 평가 테이블 작성 및 첫 번째 색을 대표로 사용
            rows = []
            if krows:
                # KMeans 결과 기반: 실제 픽셀수 사용
                for h, cnt in krows[:3]:
                    met = score_color(h, PALETTE)
                    rows.append((h, cnt, met["score"], met["nearest_delta_e"]))
            else:
                # LLM 결과 기반: 픽셀수는 미정(0)
                for h in hex_list:
                    met = score_color(h, PALETTE)
                    rows.append((h, 0, met["score"], met["nearest_delta_e"]))
            table_html = render_top_colors(rows)

            # 대표 색 선택: 첫 번째
            if hex_list:
                chosen_hex = hex_list[0]
            else:
                gr.Warning("이미지에서 유효한 색을 추출하지 못했습니다.")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            # 대표 색 기반 결과 생성
            met = score_color(chosen_hex, PALETTE)
            sw = render_swatch(chosen_hex, f"(RGB: {','.join(map(str, hex_to_rgb(chosen_hex)))})")
            score_html = render_score(met["score"], met["label"]) 
            nearest_hex, nearest_de, cat = nearest_palette(chosen_hex, PALETTE)
            nearest_html = render_nearest(nearest_hex, nearest_de, cat)
            warn_html = render_avoid_warning(met["avoid_near"]) 

            # HEX/RGB 동기값
            rr, gg, bb = hex_to_rgb(chosen_hex)
            return (
                gr.update(value=sw),
                gr.update(value=score_html),
                gr.update(value=nearest_html),
                gr.update(value=table_html),
                gr.update(value=warn_html),
                int(rr), int(gg), int(bb),
            )
        else:
            gr.Warning("유효한 HEX/RGB가 없고 이미지도 없습니다. 입력을 확인하세요.")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # 단일 색 평가 경로
        met = score_color(chosen_hex, PALETTE)
        sw = render_swatch(chosen_hex, f"(RGB: {','.join(map(str, hex_to_rgb(chosen_hex)))})")
        score_html = render_score(met["score"], met["label"]) 
        nearest_hex, nearest_de, cat = nearest_palette(chosen_hex, PALETTE)
        nearest_html = render_nearest(nearest_hex, nearest_de, cat)
        warn_html = render_avoid_warning(met["avoid_near"]) 
        # 단일 색이므로 상위 3색 표는 비움
        rr, gg, bb = hex_to_rgb(chosen_hex)
        return (
            gr.update(value=sw),
            gr.update(value=score_html),
            gr.update(value=nearest_html),
            "",  # top colors html
            gr.update(value=warn_html),
            int(rr), int(gg), int(bb),
        )
    except Exception as e:
        gr.Warning(f"분석 중 오류가 발생했습니다: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def build_ui():
    with gr.Blocks(title="봄 브라이트 컬러 매칭") as demo:
        gr.Markdown("# 봄 브라이트 컬러 매칭 (Try Spring Bright!)")
        gr.Markdown(
            "- 가이드: 그레이/멜란지/스모키/차콜 계열은 점수 하향 경향이 있습니다.\n"
            "- 입력: HEX/RGB 또는 이미지 업로드 후 분석을 눌러보세요. 이미지 클릭으로 픽셀 색을 선택할 수 있습니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                hex_in = gr.Textbox(label="HEX 입력 (#RRGGBB)", placeholder="#FF6B5C")
                with gr.Row():
                    r_in = gr.Number(label="R", value=255, precision=0)
                    g_in = gr.Number(label="G", value=107, precision=0)
                    b_in = gr.Number(label="B", value=92, precision=0)
                img_in = gr.Image(label="이미지 업로드", type="pil")
                api_key = gr.Textbox(label="Gemini API 키 (선택)", type="password")
                analyze_btn = gr.Button("분석하기")

            with gr.Column(scale=1):
                swatch_out = gr.HTML(label="선택 색상")
                score_out = gr.HTML(label="일치도 점수")
                nearest_out = gr.HTML(label="가까운 팔레트")
                top_colors_out = gr.HTML(label="상위 3색")
                warn_out = gr.HTML(label="경고")

        # 동기화 이벤트: HEX -> RGB/스와치
        hex_in.change(fn=sync_from_hex, inputs=[hex_in], outputs=[r_in, g_in, b_in, swatch_out])
        # RGB -> HEX/스와치 (각 Number 변경 시 동작)
        r_in.change(fn=sync_from_rgb, inputs=[r_in, g_in, b_in], outputs=[hex_in, swatch_out])
        g_in.change(fn=sync_from_rgb, inputs=[r_in, g_in, b_in], outputs=[hex_in, swatch_out])
        b_in.change(fn=sync_from_rgb, inputs=[r_in, g_in, b_in], outputs=[hex_in, swatch_out])

        # 이미지 클릭 스포이드
        img_in.select(fn=on_image_select, inputs=[img_in], outputs=[hex_in, r_in, g_in, b_in, swatch_out])

        # 분석 버튼
        analyze_btn.click(
            fn=analyze,
            inputs=[hex_in, r_in, g_in, b_in, img_in, api_key],
            outputs=[swatch_out, score_out, nearest_out, top_colors_out, warn_out, r_in, g_in, b_in],
        )

    return demo


if __name__ == "__main__":
    # 콘솔 로그는 한국어로만 출력
    print("Gradio 앱을 시작합니다. 브라우저에서 접속하세요.")
    app = build_ui()
    # 모바일에서도 보이도록 compact 레이아웃 유지
    app.launch()
