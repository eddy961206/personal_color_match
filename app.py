"""봄 브라이트 UX/알고리즘 업그레이드 Gradio 앱"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans

from color_utils import (
    clamp,
    hex_to_rgb,
    rgb_to_hex,
    rgb_to_hsv01,
    validate_hex,
)
from color_metrics.adjustment import ScoreBreakdown, evaluate_color
from color_metrics.recommend import suggest_alternatives
from config import DEFAULT_CONFIG, TONE_DISPLAY_ORDER
from palette_loader import PaletteRepository
from ui.colorwheel import generate_colorwheel_image
from ui.overlays import Marker, average_color, draw_markers, draw_selection_box

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("spring_bright")

REPO = PaletteRepository(Path("palettes"))
DEFAULT_TONE = TONE_DISPLAY_ORDER[0][0]

LABEL_RULES: List[Tuple[int, str]] = [
    (90, "최상"),
    (75, "좋음"),
    (60, "보통"),
    (0, "비추천"),
]

AVOID_REASON_MAP = {
    "회색기": "회색기(저채도)",
    "저채도": "회색기(저채도)",
    "저명도": "저명도(너무 어두움)",
    "쿨": "쿨 기운 과다",
    "딥": "저명도(너무 어두움)",
}


def tone_options() -> List[Tuple[str, str]]:
    return TONE_DISPLAY_ORDER


def resolve_tone_id(label: str) -> str:
    for tone_id, display in tone_options():
        if display == label:
            return tone_id
    return DEFAULT_TONE


def label_for_score(score: int) -> str:
    for threshold, label in LABEL_RULES:
        if score >= threshold:
            return label
    return "비추천"


def render_swatch(hex_color: str, rgb: Tuple[int, int, int]) -> str:
    return f"""
    <div style='display:flex;align-items:center;gap:16px;'>
      <div style='width:140px;height:140px;border-radius:16px;border:3px solid #222;background:{hex_color}'></div>
      <div style='font-size:14px;line-height:1.6'>
        <div><b>선택 색상</b></div>
        <div>HEX: <code>{hex_color}</code></div>
        <div>RGB: <code>{rgb[0]}, {rgb[1]}, {rgb[2]}</code></div>
        <div style='color:#555;font-size:13px;'>실시간으로 채점 결과가 갱신돼.</div>
      </div>
    </div>
    """


def _gauge_color(score: int) -> str:
    if score >= 90:
        return "linear-gradient(90deg,#FFB347,#FF6B5C)"
    if score >= 75:
        return "linear-gradient(90deg,#FFE08A,#FFAA5C)"
    if score >= 60:
        return "linear-gradient(90deg,#FFEFB0,#F4D35E)"
    return "linear-gradient(90deg,#F8D7DA,#E57373)"


def render_score_card(result: ScoreBreakdown) -> str:
    score = result.score
    label = label_for_score(score)
    gauge_color = _gauge_color(score)
    tooltip = "ΔE는 팔레트와의 색 차이(작을수록 유사). 채도/명도/웜 보정으로 봄 브라이트 특성을 반영."\
        " 대비 가산점은 조합 시 명암 차이를 고려해." \

    return f"""
    <div style='display:flex;flex-direction:column;gap:12px;'>
      <div style='display:flex;align-items:center;gap:12px;'>
        <div style='flex:1;height:22px;border-radius:9999px;background:#eee;overflow:hidden;position:relative;'>
          <div style='position:absolute;left:0;top:0;height:100%;width:{score}%;background:{gauge_color};transition:width 0.2s;'></div>
        </div>
        <div style='font-size:22px;font-weight:700;'>{score}</div>
        <div style='padding:4px 10px;border-radius:9999px;border:1px solid #444;' title='{tooltip}'>{label}</div>
      </div>
      <div style='font-size:13px;color:#555;'>
        ΔE 최솟값 <b>{result.delta_e:.1f}</b>, HSV 보정 <b>{result.hsv_boost:+.1f}</b>, 회피 페널티 <b>{-result.avoid_penalty:.1f}</b>, 대비 보너스 <b>{result.contrast_bonus:+.1f}</b>
      </div>
    </div>
    """


def render_nearest_info(result: ScoreBreakdown) -> str:
    nearest = result.nearest
    return f"""
    <div style='display:flex;align-items:center;gap:12px;'>
      <div style='font-weight:600;'>가장 가까운 팔레트</div>
      <div style='width:26px;height:26px;border-radius:6px;border:2px solid #111;background:{nearest['hex']}'></div>
      <div>{nearest['name']} <code>{nearest['hex']}</code> ({'베이스' if nearest['group']=='base' else '엑센트'})</div>
    </div>
    """


def avoid_message(result: ScoreBreakdown, tone_id: str) -> str:
    if not result.avoid:
        return ""
    tags = result.avoid.get("tags", []) if isinstance(result.avoid, dict) else []
    reason = next((AVOID_REASON_MAP[tag] for tag in AVOID_REASON_MAP if tag in tags), None)
    text = reason or "봄 브라이트 이미지가 탁해질 수 있어."
    return f"""
    <div style='padding:10px 14px;border-radius:12px;border:1px solid #E57373;background:#FDECEA;color:#C62828;'>
      <b>주의:</b> {result.avoid['name']}(<code>{result.avoid['hex']}</code>) 색감과 ΔE {result.avoid['distance']:.1f}로 가까워. {text}
    </div>
    """


def render_palette_panel(tone_id: str, highlight_hex: str | None, avoid_hex: str | None) -> str:
    tone = REPO.load(tone_id)
    sections = []
    for group, title, desc, icon in [
        ("base", "베이스", "피부와 가장 자연스럽게 어울리는 기본 영역", "🌤"),
        ("accent", "엑센트", "포인트로 쓰면 생기를 주는 강채도 영역", "✨"),
        ("avoid", "피해야 할", "탁색/저채도는 혈색을 가려.", "⚠️"),
    ]:
        chips = []
        for color in tone.groups.get(group, []):
            border = "4px solid #222" if highlight_hex and color.hex.upper() == highlight_hex.upper() else "1px solid #ddd"
            if avoid_hex and color.hex.upper() == avoid_hex.upper():
                border = "3px dashed #E57373"
            chips.append(
                f"""
                <div style='flex:1 0 90px;display:flex;flex-direction:column;align-items:center;gap:6px;'>
                  <div style='width:70px;height:70px;border-radius:12px;border:{border};background:{color.hex}'></div>
                  <div style='font-size:12px;font-weight:600;text-align:center;'>{color.name}</div>
                  <code style='font-size:11px'>{color.hex}</code>
                </div>
                """
            )
        sections.append(
            f"""
            <div style='flex:1;min-width:240px;padding:12px;border-radius:12px;border:1px solid #ddd;background:#fff;'>
              <div style='display:flex;align-items:center;gap:6px;font-weight:700;margin-bottom:4px;'>{icon} {title}</div>
              <div style='font-size:12px;color:#666;margin-bottom:8px;'>{desc}</div>
              <div style='display:flex;flex-wrap:wrap;gap:12px;'>{''.join(chips)}</div>
            </div>
            """
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:16px;'>" + "".join(sections) + "</div>"


def render_recommendation_section(result: ScoreBreakdown, tone, current_hex: str) -> str:
    if result.score >= 85:
        return "<div style='font-size:13px;color:#4CAF50;'>봄 브라이트 팔레트와 매우 잘 어울려! 추가 추천이 필요 없을 정도야.</div>"
    recs = suggest_alternatives(current_hex, tone)
    chips = []
    for rec in recs:
        chips.append(
            f"<span style='padding:6px 10px;border-radius:9999px;border:2px solid #FF6B5C;margin-right:8px;font-weight:600;' title='{rec.reason}'>{rec.name} <code>{rec.hex}</code></span>"
        )
    return "<div><b>이 색 대신 →</b> " + " / ".join(chips) + "</div>"


def render_help_panel() -> str:
    return """
    <div style='display:flex;flex-direction:column;gap:12px;'>
      <div><b>어떻게 점수를 올리나요?</b></div>
      <ul style='margin:0;padding-left:18px;font-size:13px;color:#444;'>
        <li>채도를 살짝 높여서 흐릿함(회색기)을 줄여봐.</li>
        <li>명도를 0.6 이상으로 끌어올리면 얼굴이 환해 보여.</li>
        <li>노랑~코랄~웜 청록 사이의 웜 기울기를 유지하면 좋고, 쿨톤으로 빠지면 점수가 낮아져.</li>
        <li>베이스 대비를 고려해서 너무 어둡거나 탁한 영역은 피하는 게 좋아.</li>
      </ul>
    </div>
    """


def _resize_longest_side(pil_img: Image.Image, longest: int = 640) -> Image.Image:
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


def extract_colors_kmeans(pil_img: Image.Image, k: int = 5):
    img = _resize_longest_side(pil_img.convert("RGB"), 512)
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_
    counts = np.bincount(labels, minlength=k)

    hsv_centers = skcolor.rgb2hsv(centers.reshape(1, k, 3))[0]
    rows = []
    coords = labels.reshape(h, w)
    for i in range(k):
        s = hsv_centers[i, 1]
        v = hsv_centers[i, 2]
        if s < 0.18 or v < 0.22:
            continue
        r, g, b = (centers[i] * 255).round().astype(int)
        hex_c = rgb_to_hex(int(r), int(g), int(b))
        ys, xs = np.where(coords == i)
        if ys.size == 0:
            cx = cy = 0
        else:
            cx = int(xs.mean())
            cy = int(ys.mean())
        rows.append({
            "hex": hex_c,
            "count": int(counts[i]),
            "centroid": (cx, cy),
        })
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows, img


def log_score(result: ScoreBreakdown):
    logger.info(
        "[정보] 점수 갱신: ΔE=%.1f, HSV보정=%+.1f → 최종 %d점",
        result.delta_e,
        result.hsv_boost / 10.0,
        result.score,
    )


def compute_score(hex_color: str, tone_id: str) -> Tuple[str, str, str, Image.Image, str, str, str]:
    tone = REPO.load(tone_id)
    breakdown = evaluate_color(hex_color, tone, REPO, DEFAULT_CONFIG)
    log_score(breakdown)
    rgb = hex_to_rgb(hex_color)
    swatch_html = render_swatch(hex_color, rgb)
    score_html = render_score_card(breakdown)
    nearest_html = render_nearest_info(breakdown)

    avoid_hex = breakdown.avoid["hex"] if breakdown.avoid else None
    avoid_html = ""
    if breakdown.avoid:
        avoid_color_obj = next((c for c in tone.groups.get("avoid", []) if c.hex == avoid_hex), None)
        if avoid_color_obj:
            breakdown.avoid.setdefault("tags", avoid_color_obj.tags)
        if breakdown.avoid.get("distance", 999) < 12:
            avoid_html = avoid_message(breakdown, tone_id)
    palette_html = render_palette_panel(tone_id, breakdown.nearest["hex"], avoid_hex)
    recommendation_html = render_recommendation_section(breakdown, tone, hex_color)
    wheel_img = generate_colorwheel_image(hex_color, tone, breakdown.nearest["hex"], avoid_hex)
    return swatch_html, score_html, nearest_html, wheel_img, palette_html, recommendation_html, avoid_html


def pack_color_outputs(
    hex_value: str,
    computed: Tuple[str, str, str, Image.Image, str, str, str],
    annotated: Optional[Image.Image] = None,
    rgb: Optional[Tuple[int, int, int]] = None,
):
    swatch_html, score_html, nearest_html, wheel_img, palette_html, recommendation_html, avoid_html = computed
    rgb_values = rgb or hex_to_rgb(hex_value)
    return (
        hex_value,
        swatch_html,
        score_html,
        nearest_html,
        wheel_img,
        palette_html,
        recommendation_html,
        avoid_html,
        annotated if annotated is not None else gr.update(),
        hex_value,
        rgb_values[0],
        rgb_values[1],
        rgb_values[2],
    )


def on_hex_change(hex_value: str, tone_id: str):
    if not validate_hex(hex_value):
        gr.Warning("HEX 형식이 올바르지 않아. 예: #FF6B5C")
        return (
            hex_value,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=None),
            gr.update(),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(),
            hex_value,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    outputs = compute_score(hex_value.upper(), tone_id)
    return pack_color_outputs(hex_value.upper(), outputs)


def on_rgb_change(r: float, g: float, b: float, tone_id: str):
    try:
        hx = rgb_to_hex(int(r), int(g), int(b))
    except ValueError as exc:
        gr.Warning(str(exc))
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=None),
            gr.update(),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return on_hex_change(hx, tone_id)


def on_tone_change(tone_id: str, hex_value: str, r: float, g: float, b: float):
    if validate_hex(hex_value):
        return on_hex_change(hex_value, tone_id)
    return on_rgb_change(r, g, b, tone_id)


def on_image_click(img: Image.Image, evt: gr.SelectData, tone_id: str):
    if img is None:
        gr.Warning("먼저 이미지를 올려줘.")
        return
    if evt is None:
        return
    x, y = int(evt.index[0]), int(evt.index[1])
    hx = rgb_to_hex(*img.convert("RGB").getpixel((x, y)))
    computed = compute_score(hx, tone_id)
    marked = draw_markers(img, [Marker(x=x, y=y, color=hex_to_rgb(hx), label="픽")])
    return pack_color_outputs(hx, computed, annotated=marked)


def on_image_upload(img: Image.Image, tone_id: str):
    if img is None:
        return gr.update()
    rows, resized = extract_colors_kmeans(img)
    markers = []
    top_html_parts = []
    for idx, row in enumerate(rows[:3], 1):
        tone = REPO.load(tone_id)
        breakdown = evaluate_color(row["hex"], tone, REPO, DEFAULT_CONFIG)
        top_html_parts.append(
            f"<tr><td>{idx}</td><td><code>{row['hex']}</code></td><td>{breakdown.score}</td><td>{breakdown.delta_e:.1f}</td><td>{row['count']}</td></tr>"
        )
        markers.append(Marker(x=row["centroid"][0], y=row["centroid"][1], color=hex_to_rgb(row["hex"]), label=f"#{idx}"))
    table_html = """
    <table style='width:100%;border-collapse:collapse;font-size:12px;'>
      <thead><tr style='border-bottom:1px solid #ccc'><th>순위</th><th>HEX</th><th>점수</th><th>ΔE</th><th>픽셀</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """.format(rows="".join(top_html_parts))
    annotated = draw_markers(resized, markers) if markers else resized
    gr.Info("상위 3색을 추출했어. 마커를 참고해봐!")
    return table_html, annotated


def on_image_select_event(img: Image.Image, evt: gr.SelectData, tone_label: str):
    tone_id = resolve_tone_id(tone_label)
    if img is None or evt is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=None),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    box = getattr(evt, "bounding_box", None) or getattr(evt, "region", None)
    if not box:
        points = getattr(evt, "selected_points", None)
        if points and len(points) > 1:
            box = {
                "x0": min(p[0] for p in points),
                "y0": min(p[1] for p in points),
                "x1": max(p[0] for p in points),
                "y1": max(p[1] for p in points),
            }
    if box:
        x0 = int(box.get("x0", box.get("x", 0)))
        y0 = int(box.get("y0", box.get("y", 0)))
        x1 = int(box.get("x1", box.get("x", 0) + box.get("width", 0)))
        y1 = int(box.get("y1", box.get("y", 0) + box.get("height", 0)))
        hx, rgb = average_color(img, x0, y0, x1, y1)
        tone = REPO.load(tone_id)
        breakdown = evaluate_color(hx, tone, REPO, DEFAULT_CONFIG)
        logger.info("[정보] 선택 영역 평균색: %s (LAB 대비 포함) → %d점", hx, breakdown.score)
        computed = compute_score(hx, tone_id)
        boxed = draw_selection_box(img, x0, y0, x1, y1)
        return pack_color_outputs(hx, computed, annotated=boxed, rgb=rgb)
    return on_image_click(img, evt, tone_id)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="봄 브라이트 컬러 매칭 업그레이드", css="body{background:#faf8f5;}") as demo:
        gr.Markdown("""
        # 봄 브라이트 컬러 매칭 실시간 체커
        따뜻함 · 채도 · 명도를 동시에 살펴보면서, 지금 선택한 색이 얼마나 봄 브라이트 톤에 어울리는지 150ms 디바운스로 즉시 확인해봐.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                tone_dropdown = gr.Dropdown(choices=[label for _, label in tone_options()], label="분석 톤", value=tone_options()[0][1], interactive=True)
                hex_box = gr.Textbox(label="HEX 입력", value="#FF6B5C", max_lines=1)
                color_picker = gr.ColorPicker(label="컬러 피커", value="#FF6B5C")
                with gr.Row():
                    r_slider = gr.Slider(0, 255, value=255, step=1, label="R")
                    g_slider = gr.Slider(0, 255, value=107, step=1, label="G")
                    b_slider = gr.Slider(0, 255, value=92, step=1, label="B")
                swatch_out = gr.HTML(label="선택 색상")
                score_out = gr.HTML(label="일치도 점수(봄 브라이트 적합도)")
                nearest_out = gr.HTML(label="가까운 팔레트")
                colorwheel_out = gr.Image(label="색상환 미니 뷰", image_mode="RGBA")
            with gr.Column(scale=1):
                palette_out = gr.HTML(label="팔레트 미리보기")
                recommendation_out = gr.HTML(label="대안 추천")
                avoid_out = gr.HTML(label="주의 안내")
                gr.HTML(render_help_panel(), label="가이드")
        with gr.Row():
            with gr.Column():
                upload = gr.Image(label="이미지 업로드", type="pil")
                analysis_table = gr.HTML(label="추출 색상 점수표")
            with gr.Column():
                annotated = gr.Image(label="마커/드래그 결과", type="pil")
        gr.Markdown("※ 이미지 위를 드래그하면 선택 영역 평균색을 계산해.")

        # 초기 렌더링
        initial_outputs = compute_score("#FF6B5C", DEFAULT_TONE)
        swatch_out.value, score_out.value, nearest_out.value, colorwheel_out.value, palette_out.value, recommendation_out.value, avoid_out.value = initial_outputs
        analysis_table.value = ""
        annotated.value = None

        # 이벤트 연결
        tone_dropdown.change(
            fn=lambda tone_label, hex_value, r, g, b: on_tone_change(resolve_tone_id(tone_label), hex_value, r, g, b),
            inputs=[tone_dropdown, hex_box, r_slider, g_slider, b_slider],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
        )

        hex_box.input(
            fn=lambda hx, tone_label: on_hex_change(hx, resolve_tone_id(tone_label)),
            inputs=[hex_box, tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
            debounce=0.15,
        )
        color_picker.input(
            fn=lambda hx, tone_label: on_hex_change(hx, resolve_tone_id(tone_label)),
            inputs=[color_picker, tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
            debounce=0.15,
        )
        r_slider.input(
            fn=lambda r, g, b, tone_label: on_rgb_change(r, g, b, resolve_tone_id(tone_label)),
            inputs=[r_slider, g_slider, b_slider, tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
            debounce=0.15,
        )
        g_slider.input(
            fn=lambda r, g, b, tone_label: on_rgb_change(r, g, b, resolve_tone_id(tone_label)),
            inputs=[r_slider, g_slider, b_slider, tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
            debounce=0.15,
        )
        b_slider.input(
            fn=lambda r, g, b, tone_label: on_rgb_change(r, g, b, resolve_tone_id(tone_label)),
            inputs=[r_slider, g_slider, b_slider, tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
            debounce=0.15,
        )

        upload.change(
            fn=lambda img, tone_label: on_image_upload(img, resolve_tone_id(tone_label)),
            inputs=[upload, tone_dropdown],
            outputs=[analysis_table, annotated],
        )
        upload.select(
            fn=on_image_select_event,
            inputs=[upload, gr.EventData(), tone_dropdown],
            outputs=[hex_box, swatch_out, score_out, nearest_out, colorwheel_out, palette_out, recommendation_out, avoid_out, annotated, color_picker, r_slider, g_slider, b_slider],
        )

    return demo


if __name__ == "__main__":
    print("Gradio 앱을 시작할게. 브라우저에서 확인해줘.")
    app = build_ui()
    app.launch()
