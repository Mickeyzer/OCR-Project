import cv2
import numpy as np
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

# Path to the numeric-only PaddleOCR dictionary
DICT_PATH = str(Path(__file__).parent / "numeric_dict.txt")

# Singleton OCR engine (loaded once, reused on every inference call)
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(
                use_angle_cls=False,    # Digits are always horizontal
                lang='en',
                rec_char_dict_path=DICT_PATH,   # Restrict to 0-9 only
                use_gpu=False,          # CPU deployment
                show_log=False
            )
            logger.info("PaddleOCR engine loaded for Rolling Drum pipeline.")
        except ImportError:
            logger.error("PaddleOCR is not installed.")
            _ocr_engine = None
    return _ocr_engine


# ---------------------------------------------------------
# 1. ROI Extraction — Digit Window
# ---------------------------------------------------------

def extract_digit_window(image: np.ndarray) -> np.ndarray:
    """
    Extracts only the rectangular digit window from the broader meter crop.
    Uses morphological closing to merge closely packed digit slots into a
    single contiguous blob, then finds the widest, most rectangular contour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vertical Sobel to highlight digit edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))

    # Threshold
    _, thresh = cv2.threshold(sobel_y, 40, 255, cv2.THRESH_BINARY)

    # Wide morphological closing: merges closely-spaced digit slots into one blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No digit window contour found. Returning full crop.")
        return image

    h_img, w_img = image.shape[:2]

    # Filter contours by aspect ratio (digit windows are wide, not tall)
    best = None
    best_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        area = w * h
        # Digit window: aspect ratio 2:1 to 8:1, covers at least 20% of image width
        if 2.0 <= aspect <= 8.0 and w > w_img * 0.2 and area > best_area:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        logger.warning("No suitable digit window contour found. Returning full crop.")
        return image

    x, y, w, h = best
    # Add a small padding to avoid clipping digit edges
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    return image[y1:y2, x1:x2]


# ---------------------------------------------------------
# 2. Deskewing
# ---------------------------------------------------------

def deskew_window(image: np.ndarray) -> np.ndarray:
    """
    Detects rotation angle using minAreaRect on the largest contour
    and corrects it with warpAffine. Skips correction for angles < 1 degree.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]

    # minAreaRect returns angle in [-90, 0). Normalize to meaningful skew angle.
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 1.0:
        return image  # Not worth rotating for tiny angles

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.debug(f"Deskewed by {angle:.2f} degrees.")
    return deskewed


# ---------------------------------------------------------
# 3. Preprocessing — Bilateral + Top-Hat + Adaptive Threshold
# ---------------------------------------------------------

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Specialized preprocessing for rolling drums:
    1. Bilateral Filter     — removes glass glare, keeps digit edges sharp
    2. CLAHE                — forces faded, worn digits to stand out
    3. Top-Hat Transform    — neutralizes deep shadows between drum slots
    4. Adaptive Threshold   — handles uneven lighting across the digit window
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Bilateral filter to remove glare while preserving sharp digit edges
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE — critical for mechanically worn drums with faded paint
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(filtered)

    # 3. Top-Hat Transform — subtracts locally dark shadows between slots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # Add Top-Hat back to enhanced to brighten digit strokes
    combined = cv2.add(enhanced, tophat)

    # 4. Gaussian Adaptive Threshold — handles shadow gradients across slots
    thresh = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8
    )

    # Convert back to BGR so PaddleOCR receives its expected 3-channel input
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------
# 4. OCR Inference + Character-Level Confidence Filtering
# ---------------------------------------------------------

def run_ocr_and_filter(preprocessed_img: np.ndarray) -> Tuple[str, float]:
    """
    Runs PaddleOCR on the preprocessed drum image.
    - Applies character-level confidence filtering (threshold: 0.70).
      Low-confidence characters are replaced with '?' to signal partial reads.
    - Returns (reading_string, mean_confidence).
    """
    engine = get_ocr_engine()
    if engine is None:
        raise RuntimeError("PaddleOCR engine not available.")

    result = engine.ocr(preprocessed_img, cls=False)

    if not result or not result[0]:
        raise ValueError("PaddleOCR returned no text from the digit window.")

    # Collect all text segments (PaddleOCR may split the window into multiple lines)
    all_chars: List[str] = []
    all_confidences: List[float] = []

    for line in result[0]:
        text: str = line[1][0]
        conf: float = line[1][1]

        # PaddleOCR returns word-level confidence; use it as proxy per-character
        # (Character-level confidence requires accessing internals — we use mean here)
        for char in text:
            all_chars.append(char if conf >= 0.70 else "?")
            all_confidences.append(conf)

    reading = "".join(all_chars)
    mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

    return reading, mean_conf


# ---------------------------------------------------------
# 5. Post-Processing Rules for Partially Rolled Digits
# ---------------------------------------------------------

def post_process_reading(raw_reading: str) -> str:
    """
    Applies the Lower-Bound Rule for '?' characters introduced by 
    low-confidence half-rolled digits.

    Rules:
      Rule 1: If reading ends in '?', replace with '0' (lower-bound safety).
    """
    if "?" not in raw_reading:
        return raw_reading

    # Rule 1: Replace trailing '?' with '0' (lower-bound billing rule)
    processed = re.sub(r'\?+$', lambda m: '0' * len(m.group()), raw_reading)

    return processed


# ---------------------------------------------------------
# 6. Master Orchestrator
# ---------------------------------------------------------

def get_drum_reading(
    image: np.ndarray
) -> Tuple[str, float]:
    """
    Master function for the Rolling Drum pipeline.
    Returns (reading_string, confidence).
    Raises an exception on unrecoverable failure (caught by the router).
    """
    # Step 1: Extract digit window
    digit_window = extract_digit_window(image)

    # Step 2: Deskew
    deskewed = deskew_window(digit_window)

    # Step 3: Preprocess
    preprocessed = preprocess_for_ocr(deskewed)

    # Step 4: OCR
    raw_reading, confidence = run_ocr_and_filter(preprocessed)
    logger.info(f"Raw OCR reading: '{raw_reading}' | Confidence: {confidence:.3f}")

    # Step 5: Post-process
    final_reading = post_process_reading(raw_reading)
    logger.info(f"Final reading after post-processing: '{final_reading}'")

    return final_reading, confidence
