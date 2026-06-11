import os
import random
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Configuration
NUM_IMAGES = 5000
IMAGE_DIR = "digital_dataset/images"
LABEL_FILE = "digital_dataset/train_labels.txt"
FONT_URL = "https://github.com/keshikan/DSEG/raw/master/fonts/DSEG7-Classic/DSEG7Classic-Regular.ttf"
FONT_PATH = "DSEG7Classic-Regular.ttf"

def download_font():
    if not os.path.exists(FONT_PATH):
        print("Downloading 7-segment font...")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        print("Font downloaded.")

def generate_random_digital_string():
    """Generates strings like '12.34', '-0.5', '888', etc."""
    is_negative = random.random() < 0.2
    has_decimal = random.random() < 0.8
    
    num_digits = random.randint(1, 5)
    digits = "".join([str(random.randint(0, 9)) for _ in range(num_digits)])
    
    if has_decimal and len(digits) > 1:
        dec_pos = random.randint(1, len(digits) - 1)
        digits = digits[:dec_pos] + "." + digits[dec_pos:]
        
    if is_negative:
        digits = "-" + digits
        
    return digits

def add_broken_segments(img_np: np.ndarray) -> np.ndarray:
    """Randomly draws thin black lines through the white text to simulate broken LCD bars."""
    h, w = img_np.shape[:2]
    num_lines = random.randint(0, 3)
    
    for _ in range(num_lines):
        # Draw a horizontal or vertical black line
        if random.random() < 0.5:
            # Horizontal line
            y = random.randint(0, h)
            thickness = random.randint(1, 3)
            cv2.line(img_np, (0, y), (w, y), (0, 0, 0), thickness)
        else:
            # Vertical line
            x = random.randint(0, w)
            thickness = random.randint(1, 3)
            cv2.line(img_np, (x, 0), (x, h), (0, 0, 0), thickness)
            
    return img_np

def add_glare_and_blur(img_np: np.ndarray) -> np.ndarray:
    """Simulates motion blur and screen glare."""
    # Motion Blur
    if random.random() < 0.3:
        k_size = random.choice([3, 5])
        img_np = cv2.blur(img_np, (k_size, k_size))
        
    # Glare (Overlay a faint white polygon)
    if random.random() < 0.3:
        overlay = img_np.copy()
        h, w = img_np.shape[:2]
        pts = np.array([
            [random.randint(0, w), 0],
            [w, 0],
            [w, random.randint(0, h)],
            [random.randint(0, w), h]
        ], np.int32)
        cv2.fillPoly(overlay, [pts], (100, 100, 100))
        img_np = cv2.addWeighted(overlay, 0.3, img_np, 0.7, 0)
        
    return img_np

def generate_dataset():
    download_font()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    font = ImageFont.truetype(FONT_PATH, size=48)
    
    print(f"Generating {NUM_IMAGES} synthetic images...")
    
    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        for i in range(NUM_IMAGES):
            text = generate_random_digital_string()
            
            # Create a blank black image
            img_pil = Image.new('RGB', (200, 80), color=(0, 0, 0))
            draw = ImageDraw.Draw(img_pil)
            
            # Draw text in white
            # To add realism, slightly offset the text
            offset_x = random.randint(5, 20)
            offset_y = random.randint(5, 20)
            draw.text((offset_x, offset_y), text, font=font, fill=(255, 255, 255))
            
            # Convert PIL to OpenCV numpy array for augmentations
            img_np = np.array(img_pil)
            
            # Apply Augmentations
            img_np = add_broken_segments(img_np)
            img_np = add_glare_and_blur(img_np)
            
            # Tight crop around the text to mimic MobileNet bounding box
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Add random padding
                px, py = random.randint(2, 8), random.randint(2, 8)
                img_np = img_np[max(0, y-py):y+h+py, max(0, x-px):x+w+px]
            
            # Save
            filename = f"digital_{i:05d}.jpg"
            filepath = os.path.join(IMAGE_DIR, filename)
            cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # Write label (PaddleOCR format: path/to/image.jpg\tLABEL)
            f.write(f"images/{filename}\t{text}\n")
            
            if (i+1) % 500 == 0:
                print(f"Generated {i+1}/{NUM_IMAGES}")
                
    print("Dataset generation complete!")
    print(f"Images saved to: {IMAGE_DIR}")
    print(f"Labels saved to: {LABEL_FILE}")

if __name__ == "__main__":
    generate_dataset()
