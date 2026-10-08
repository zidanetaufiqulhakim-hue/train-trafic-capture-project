"""
Image preprocessing and OCR text extraction for individual bounding box crops.

preprocess_crop() and read_text() are the only public functions.
"""

import cv2
import numpy as np
import easyocr


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Prepares a detected crop for EasyOCR.

    Steps (order matters):
      1. Grayscale — removes color noise, reduces input dimensions
      2. 2× upscale with INTER_CUBIC — EasyOCR reads small text much better at larger sizes
      3. CLAHE contrast enhancement — sharpens edges without adding ringing artifacts
      4. Back to BGR — EasyOCR accepts BGR input

    CLAHE is used instead of a sharpening kernel because kernel-based sharpening
    amplifies compression noise in screen recordings."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def read_text(crop: np.ndarray, reader: easyocr.Reader) -> str:
    """Runs EasyOCR on a preprocessed crop. Returns joined text (may be empty string)."""
    results = reader.readtext(crop, detail=0)
    return " ".join(results) if results else ""
