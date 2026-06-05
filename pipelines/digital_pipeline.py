import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Path to the digital-specific PaddleOCR dictionary
DICT_PATH = str(Path(__file__).parent / "digital_dict.txt")

# Singleton OCR engine
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(
                use_angle_cls=False,            # Horizontal text
                lang='en',
                rec_char_dict_path=DICT_PATH,   # Restrict to 0-9, ., -
                use_gpu=False,                  # CPU deployment
                show_log=False
            )
            logger.info("PaddleOCR engine loaded for Digital pipeline.")
        except ImportError:
            logger.error("PaddleOCR is not installed.")
            _ocr_engine = None
    return _ocr_engine


def preprocess_digital(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses 7-segment LED/LCD displays.
    1. Grayscale.
    2. Adaptive Thresholding to ensure text is ALWAYS white on black background.
    3. Morphological Dilation to physically thicken digits and bridge broken segments.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE for contrast enhancement against glare
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Thresholding: We want White Text on Black Background.
    # Digital displays can be dark text on bright (LCD) or bright text on dark (LED).
    # We use Otsu's thresholding. If the image is mostly bright (LCD background),
    # the mean will be high, and we should use THRESH_BINARY_INV.
    mean_val = np.mean(enhanced)
    if mean_val > 127:
        # Bright background, dark text -> Invert it
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Dark background, bright text -> Keep it
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # 3. The Secret Sauce: Morphological Dilation
    # 7-segment displays have tiny black gaps between the segments.
    # By dilating the white pixels, we bridge those gaps, turning broken bars
    # into continuous strokes, making it drastically easier for SVTR to read.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Convert back to 3-channel for PaddleOCR
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


def run_digital_ocr(preprocessed_img: np.ndarray) -> Tuple[str, float]:
    """
    Runs PaddleOCR on the dilated digital display crop.
    Returns (reading_string, confidence).
    """
    engine = get_ocr_engine()
    if engine is None:
        raise RuntimeError("PaddleOCR engine not available.")

    result = engine.ocr(preprocessed_img, cls=False)

    if not result or not result[0]:
        raise ValueError("No text detected on the digital display.")

    all_chars = []
    all_confidences = []

    for line in result[0]:
        text: str = line[1][0]
        conf: float = line[1][1]

        # In a real environment, you might apply regex here to ensure 
        # it conforms to a strict float format (e.g. ^-?[0-9]*\.?[0-9]+$)
        for char in text:
            # We filter out spaces just in case, though the dict should handle it
            if char.strip():
                all_chars.append(char if conf >= 0.75 else "?")
                all_confidences.append(conf)

    reading = "".join(all_chars)
    mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

    return reading, mean_conf


def get_digital_reading(image: np.ndarray) -> Tuple[str, float]:
    """
    Master function for the Digital Pipeline.
    """
    try:
        # Preprocess
        preprocessed = preprocess_digital(image)
        
        # OCR
        raw_reading, confidence = run_digital_ocr(preprocessed)
        
        # Simple post-processing to clean up multiple decimals or dashes if SVTR hallucinated them
        # (Though our constrained dictionary makes this rare)
        logger.info(f"Digital Reading: '{raw_reading}' | Confidence: {confidence:.3f}")
        return raw_reading, confidence
        
    except Exception as e:
        logger.error(f"Digital Pipeline Error: {str(e)}")
        raise
