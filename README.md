# Automated Meter Reading (AMR) Pipeline

A highly modular, purely CPU-optimized computer vision pipeline for reading three distinct types of industrial meters: **Analog**, **Rolling Drum**, and **Digital (7-Segment)**.

## Overview

This project uses classical OpenCV preprocessing to neutralize environmental noise (glare, shadows, perspective skew) before passing the cleaned images to a lightweight PaddleOCR (SVTR) engine for rapid inference. The architecture relies on the **Strategy Pattern** to dynamically route images to the correct specialized pipeline.

## Architecture & Features

### 1. The Router (`amr_router.py`)
The central hub of the application. It receives a cropped image, instantiates the correct reading strategy, and returns a standardized Pydantic `ReadingResult` object containing the extracted value, confidence score, and status, ready for database insertion.

### 2. Analog Pipeline (`pipelines/analog_pipeline.py`)
*   **Challenge:** Hough Lines fail due to reflections and shadows.
*   **Solution:** Uses **Polar Unwrapping** to convert the circular dial into a flat rectangle, transforming the curved numbers into a straight line and the needle into a thick vertical line. It then uses **Dynamic OCR Interpolation** to mathematically calculate the gauge scale on the fly without needing configuration files.

### 3. Rolling Drum Pipeline (`pipelines/drum_pipeline.py`)
*   **Challenge:** Deep mechanical shadows between digit slots ruin thresholding, and half-rolled digits confuse standard OCR.
*   **Solution:** Extracts the digit window and applies a sequence of **Bilateral Filtering**, **CLAHE**, and **Morphological Top-Hat Transforms** to subtract shadows. It restricts PaddleOCR to predict only digits (`0-9`) using a custom dictionary, replacing blurry half-digits with a safe lower-bound fallback.

### 4. Digital Pipeline (`pipelines/digital_pipeline.py`)
*   **Challenge:** 7-segment displays (LCD/LED) are made of disconnected bars. If glare hides a single bar, an `8` becomes a `0`.
*   **Solution:** Detects whether the display is LCD or LED and automatically inverts it to ensure white text on a black background. Applies **Morphological Dilation** to physically expand the white pixels, successfully bridging the broken microscopic gaps in the 7-segment numbers before sending the image to the OCR engine.

## Dependencies
*   `paddleocr`
*   `paddlepaddle`
*   `opencv-python`
*   `pydantic`
*   `numpy`
