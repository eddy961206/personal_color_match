# 봄 브라이트 컬러 매칭 (Try Spring Bright!)

간단한 퍼스널컬러 판별 MVP 웹앱입니다. 봄 브라이트(Spring Bright) 톤에 맞는지 점수로 보여주고, 가까운 팔레트 색상을 안내합니다. 이미지 업로드 후 자동 색 추출(KMeans)과 이미지 클릭 스포이드도 지원합니다. Gemini API 키가 있으면 LLM으로 주요 색상 3개를 자동 추출할 수 있습니다.

## 설치 및 실행

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 선택: 환경변수로 Gemini API 키 설정 (UI 입력이 우선)
# macOS/Linux:
export GEMINI_API_KEY=your_key
# Windows PowerShell:
# $env:GEMINI_API_KEY="your_key"
# Windows CMD:
# set GEMINI_API_KEY=your_key

python app.py
```

## 사용법

- 좌측에서 HEX/RGB를 입력하거나 이미지를 업로드합니다.
- 이미지를 클릭하면 해당 픽셀의 색상이 자동으로 채워집니다.
- 분석하기를 누르면 점수(0~100), 판정 라벨, 가까운 팔레트 색상을 확인할 수 있습니다.
- 이미지 업로드 시 상위 3색 후보와 각 점수를 표로 보여줍니다.
- 회색/멜란지/스모키/차콜 계열은 점수가 낮게 나올 수 있습니다.

## 기능 요약

- 입력: HEX/RGB, 이미지 업로드, 이미지 클릭 스포이드
- 자동 추출: KMeans(n=3~5) + 저채도/저밝기 필터링
- 점수: CIELAB + DeltaE (CIEDE2000) + HSV 보정
- LLM(선택): Gemini 사용 시 주요 색상 3개를 JSON으로 받음 (키 없으면 KMeans 폴백)

## 참고

- 모든 주석/콘솔 로그는 한국어로 작성되어 있습니다.
- 팔레트는 `palette.json`에서 수정/확장할 수 있습니다.
- LLM Provider는 `llm_providers.py`에 구현되어 있으며 확장 가능합니다.

