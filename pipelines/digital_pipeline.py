import cv2
import re
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Singleton OCR engine
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            # Using default PaddleOCR English model — no custom training needed.
            # Post-processing regex handles stripping invalid characters.
            _ocr_engine = PaddleOCR(
                use_angle_cls=False,  # Horizontal text only
                lang='en',
                use_gpu=False,        # CPU deployment
                show_log=False
            )
            logger.info("PaddleOCR engine loaded for Digital pipeline.")
        except ImportError:
            logger.error("PaddleOCR is not installed. Run: pip install paddleocr")
            _ocr_engine = None
    return _ocr_engine


def preprocess_digital(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses 7-segment LED/LCD displays.
    1. Grayscale + CLAHE for glare compensation.
    2. Auto-inversion: ensures text is always WHITE on BLACK.
    3. Morphological Dilation to bridge broken/faint segments.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement against glare
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Auto-inversion: LCD displays have dark text on bright backgrounds,
    # LED displays have bright text on dark backgrounds.
    mean_val = np.mean(enhanced)
    if mean_val > 127:
        # Bright background (LCD) → invert so text is white on black
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Dark background (LED) → keep as-is
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Dilation: bridges the tiny gaps between 7-segment bars,
    # turning broken strokes into continuous ones for better OCR accuracy.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Convert back to 3-channel for PaddleOCR
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


def clean_digital_reading(raw: str) -> str:
    """
    Post-processing: keep only valid 7-segment characters (digits, dot, minus).
    Removes letters and other OCR noise introduced by the general-purpose model.
    """
    # Keep only digits, a single decimal point, and a leading minus sign
    cleaned = re.sub(r'[^0-9.\-]', '', raw)
    # Collapse multiple dots or dashes (OCR artefacts)
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'\-{2,}', '-', cleaned)
    return cleaned.strip()


def run_digital_ocr(preprocessed_img: np.ndarray) -> Tuple[str, float]:
    """
    Runs PaddleOCR on the preprocessed digital display crop.
    Returns (reading_string, mean_confidence).
    """
    engine = get_ocr_engine()
    if engine is None:
        raise RuntimeError("PaddleOCR engine not available.")

    result = engine.ocr(preprocessed_img, cls=False)

    if not result or not result[0]:
        raise ValueError("No text detected on the digital display.")

    all_text = []
    all_confidences = []

    for line in result[0]:
        text: str = line[1][0]
        conf: float = line[1][1]
        all_text.append(text)
        all_confidences.append(conf)

    raw_reading = " ".join(all_text)
    mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

    return raw_reading, mean_conf


def get_digital_reading(image: np.ndarray) -> Tuple[str, float]:
    """
    Master function for the Digital Pipeline.
    Returns (cleaned_reading, confidence).
    """
    try:
        preprocessed = preprocess_digital(image)
        raw_reading, confidence = run_digital_ocr(preprocessed)
        reading = clean_digital_reading(raw_reading)

        logger.info(f"Digital Reading: '{reading}' (raw: '{raw_reading}') | Confidence: {confidence:.3f}")
        return reading, confidence

    except Exception as e:
        logger.error(f"Digital Pipeline Error: {str(e)}")
        raise
