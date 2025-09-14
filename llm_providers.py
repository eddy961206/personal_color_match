"""
LLM Provider 어댑터

모든 주석/로그는 한국어로 작성합니다.
현재는 Gemini 전용 구현을 제공하며, 추후 OpenAI/Claude 확장 가능하도록 인터페이스를 단순화합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import io
import json
import re


class ColorProvider:
    """Provider 인터페이스: 이미지에서 주요 의복 색상 HEX 목록을 추출한다."""

    def get_main_colors(self, pil_image) -> List[str]:  # pragma: no cover - 간단 인터페이스
        raise NotImplementedError


@dataclass
class GeminiProvider(ColorProvider):
    """
    Google Gemini를 사용하여 이미지에서 의복 주요 색상 3개(HEX)를 JSON으로 추출.
    실패 시 예외를 던지고, 상위 레이어에서 KMeans로 폴백하도록 한다.
    """

    api_key: str
    model_name: str = "gemini-2.5-pro"

    def _pil_to_bytes(self, pil_image) -> bytes:
        # PIL 이미지를 PNG 바이트로 변환 (LLM에 안전 전달)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()

    def _extract_hex_list(self, text: str) -> List[str]:
        # 응답에서 JSON 시도, 실패 시 HEX 정규표현식으로 추출
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # {"colors": ["#AABBCC", ...]} 형태 지원
                if "colors" in data and isinstance(data["colors"], list):
                    return [str(x).strip() for x in data["colors"]]
            if isinstance(data, list):
                return [str(x).strip() for x in data]
        except Exception:
            pass
        # JSON 파싱 실패 시, HEX 패턴 추출
        return re.findall(r"#[0-9A-Fa-f]{6}", text)

    def get_main_colors(self, pil_image) -> List[str]:
        # API 키는 절대 로그에 출력하지 않는다.
        try:
            import google.generativeai as genai
        except Exception as e:
            # 라이브러리 미설치/불가 등
            raise RuntimeError("Gemini 라이브러리를 불러오지 못했습니다: " + str(e))

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            prompt = (
                "당신은 이미지에서 의복의 주요 색상 3개를 추출하는 도우미입니다. "
                "배경이나 그림자, 피부색은 제외하고, 의복이나 액세서리의 대표 색상을 선택하세요. "
                "결과는 JSON 배열로만 출력하세요. 예: [\"#FF6B5C\", \"#6DCFF6\", \"#FFC107\"]"
            )

            img_bytes = self._pil_to_bytes(pil_image)
            # 이미지 + 프롬프트 동시 전달
            resp = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_bytes},
            ])
            text = (resp.text or "").strip()
            colors = self._extract_hex_list(text)
            # 상위 3개만 반환
            uniq = []
            for c in colors:
                if c not in uniq:
                    uniq.append(c)
            if not uniq:
                raise RuntimeError("LLM 응답에서 유효한 HEX를 찾지 못했습니다")
            return uniq[:3]
        except Exception as e:
            # 한국어 에러 메시지로 래핑하여 상위 레이어에 전달
            raise RuntimeError(f"Gemini 색상 추출 실패: {e}")

