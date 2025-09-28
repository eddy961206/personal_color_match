# Gemini Context: Personal Color Matcher

## Project Overview

This is a Python web application that acts as a "Personal Color" matcher, specifically for the "Spring Bright" color palette. It's built with Gradio and allows users to check if a specific color matches the Spring Bright tone.

The application takes a color as input (HEX, RGB, or from an image) and calculates a score from 0 to 100 based on its similarity to the predefined Spring Bright color palette.

- **Core Logic:** The main application logic is in `app.py`. It handles color conversions, scoring, and the Gradio UI definition.
- **Color Scoring:** The scoring algorithm uses the DeltaE CIEDE2000 formula to calculate the perceptual difference between the input color and the palette colors defined in `palette.json`. It also applies adjustments based on HSV values to favor bright and vivid colors.
- **Image Color Extraction:** If an image is uploaded, the app extracts dominant colors. It can use the Gemini API (if a key is provided) for intelligent extraction or fall back to a KMeans clustering algorithm. The image processing logic is in `app.py`, and the Gemini integration is in `llm_providers.py`.
- **Palette:** The reference "Spring Bright" colors are stored in `palette.json`, separated into `base`, `accent`, and `avoid` categories.

## Building and Running

The project is a standard Python application.

1.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

4.  **(Optional) Use Gemini for image analysis:**
    To enable color extraction from images using the Gemini API, set your API key as an environment variable or enter it in the UI.
    ```bash
    export GEMINI_API_KEY='your_api_key'
    ```

The application will be available at a local URL provided by Gradio upon startup.

## Development Conventions

- **Language:** The codebase, including all comments and console logs, is written entirely in Korean.
- **Modularity:**
    - The main Gradio app is in `app.py`.
    - LLM integration is abstracted in `llm_providers.py`, with a `GeminiProvider` implementation. This is designed to be extensible for other providers.
    - The color palette is managed externally in `palette.json`, making it easy to modify without changing the code.
- **Dependencies:** All Python dependencies are listed in `requirements.txt`.
- **Error Handling:** The application provides user-friendly warnings in the UI for invalid inputs or processing errors (e.g., invalid HEX code, Gemini API failure). When the LLM fails, it automatically falls back to the KMeans method.
