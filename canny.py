import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def sharpness_improve(image):
    # 흑백으로 변환
    gray = cv2.cvtColor(np.array(image) , cv2.COLOR_BGR2GRAY)
    # 이미지 선명도 올리기
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)