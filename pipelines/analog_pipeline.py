import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Note: In a production environment, you'd initialize PaddleOCR once globally
# to avoid loading the model weights on every single function call.
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            # Using english language; use_angle_cls=False since text is already unwrapped
            _ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        except ImportError:
            logger.warning("PaddleOCR not installed. OCR will fail.")
            _ocr_engine = None
    return _ocr_engine


def preprocess_and_deskew(image: np.ndarray) -> np.ndarray:
    """
    Applies bilateral filtering and CLAHE.
    (Deskewing via perspective warp is omitted here for brevity, assuming crops are decent.
    If extreme angles are common, contour detection + warpPerspective goes here.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter removes shadows but keeps the sharp edge of the needle
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # CLAHE for localized contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    return enhanced


def unwrap_dial(enhanced_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[int]]:
    """
    Uses HoughCircles to find the CENTER of the dial, and applies warpPolar.
    Returns the unwrapped image, center point, and radius.
    """
    height, width = enhanced_img.shape
    
    # We use HoughCircles purely to find the (X, Y) center point of the gauge.
    # We need the center so warpPolar knows where to unwrap from.
    circles = cv2.HoughCircles(
        enhanced_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=50, 
        param2=30, 
        minRadius=int(min(height, width) * 0.25),
        maxRadius=int(min(height, width) * 0.5)
    )
    
    if circles is None:
        return None, None, None
        
    circles = np.round(circles[0, :]).astype("int")
    # Take the strongest circle found
    cx, cy, radius = circles[0]
    
    # Calculate dimensions for the polar image
    # The X-axis will represent the angle (0 to 360 degrees) -> 360 pixels wide
    # The Y-axis will represent the radius -> 'radius' pixels high
    polar_width = 360
    polar_height = radius
    
    unwrapped = cv2.warpPolar(
        enhanced_img, 
        (polar_height, polar_width), 
        (cx, cy), 
        radius, 
        cv2.WARP_POLAR_LINEAR
    )
    
    # warpPolar returns image with Y=angle, X=radius. Let's transpose it so X=angle, Y=radius
    unwrapped = cv2.transpose(unwrapped)
    
    return unwrapped, (cx, cy), radius


def detect_needle(unwrapped_img: np.ndarray) -> Optional[float]:
    """
    Finds the needle in the polar image using vertical Sobel and column summation.
    Returns the needle angle (0.0 to 360.0).
    """
    # 1. Apply vertical Sobel to highlight vertical lines (which is what the needle became)
    sobel_x = cv2.Sobel(unwrapped_img, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
    
    # 2. Threshold to remove noise
    _, thresh = cv2.threshold(sobel_x, 100, 255, cv2.THRESH_BINARY)
    
    # 3. Sum the columns (Y-axis projection)
    column_sums = np.sum(thresh, axis=0)
    
    if np.max(column_sums) == 0:
        return None # Needle not found
        
    # 4. The column with the highest sum is the needle's exact angle.
    # Since polar_width is 360, the x-index perfectly matches the degree.
    needle_angle_deg = float(np.argmax(column_sums))
    
    return needle_angle_deg


def dynamic_ocr_scale(unwrapped_img: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Runs PaddleOCR on the unwrapped image to dynamically calculate the scale.
    Returns: (Scale_Units_Per_Degree, Anchor_Angle, Anchor_Value)
    """
    engine = get_ocr_engine()
    if engine is None:
        raise Exception("PaddleOCR is not loaded.")
        
    # Run OCR on the unwrapped image
    # Note: the unwrapped image might need slight resizing or binarization for optimal OCR
    result = engine.ocr(unwrapped_img, cls=False)
    
    if not result or not result[0]:
         raise Exception("No text detected on the dial.")
         
    detected_numbers = []
    
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        
        # We only care about highly confident numbers
        if confidence > 0.8 and text.replace('.','',1).isdigit():
            # Get the center X coordinate of the bounding box (this is the angle!)
            box_np = np.array(box)
            center_x = np.mean(box_np[:, 0])
            
            detected_numbers.append({
                'angle': center_x,
                'value': float(text)
            })
            
    # We need at least two numbers to calculate a linear scale
    if len(detected_numbers) < 2:
        raise Exception(f"Found {len(detected_numbers)} numbers. Need at least 2 for linear interpolation.")
        
    # Sort by angle (left to right in the unwrapped image)
    detected_numbers.sort(key=lambda x: x['angle'])
    
    # Pick any two numbers (we'll use the first two for simplicity)
    pt1 = detected_numbers[0]
    pt2 = detected_numbers[1]
    
    angle_diff = abs(pt2['angle'] - pt1['angle'])
    val_diff = abs(pt2['value'] - pt1['value'])
    
    if angle_diff == 0:
        raise Exception("Invalid scale calculation (divide by zero).")
        
    units_per_degree = val_diff / angle_diff
    
    return units_per_degree, pt1['angle'], pt1['value']


def get_analog_reading(image: np.ndarray) -> Tuple[Optional[float], float]:
    """
    Master function: Returns (Reading, Confidence)
    """
    try:
        # 1. Preprocess
        enhanced = preprocess_and_deskew(image)
        
        # 2. Unwrap
        unwrapped, center, radius = unwrap_dial(enhanced)
        if unwrapped is None:
            raise Exception("Failed to detect dial center. Image may be too skewed or occluded.")
            
        # 3. Detect Needle
        needle_angle = detect_needle(unwrapped)
        if needle_angle is None:
            raise Exception("Needle not found in polar projection.")
            
        # 4. OCR Scale calculation
        try:
            scale, anchor_angle, anchor_value = dynamic_ocr_scale(unwrapped)
            
            # 5. Final Linear Interpolation!
            reading = anchor_value + ((needle_angle - anchor_angle) * scale)
            
            # Since this succeeded without a hitch, confidence is high
            return float(reading), 0.95
            
        except Exception as ocr_e:
            # OCR failed (needle occluding, poor lighting, etc)
            raise Exception(f"OCR Scale detection failed: {str(ocr_e)}")
            
    except Exception as e:
        logger.error(f"Analog Pipeline Error: {str(e)}")
        # Raise it so the router catches it and marks it FAILED
        raise
