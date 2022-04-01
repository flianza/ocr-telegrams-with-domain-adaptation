import cv2
import numpy as np


def get_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9
    )

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Buscar el rectangulo de mayor area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    min_area_rect = cv2.minAreaRect(cnts[0])

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle -= 90
    return -1.0 * angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    image_ = image.copy()
    h, w = image_.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_ = cv2.warpAffine(
        image_, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return image_


def deskew(image: np.ndarray) -> np.ndarray:
    angle = get_skew_angle(image)

    # si da un valor verdura, mejor no hacemos nada
    if abs(angle) > 30:
        return image

    return rotate_image(image, -1.0 * angle)
