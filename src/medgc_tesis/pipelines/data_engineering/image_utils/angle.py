import cv2
import numpy as np


def get_skew_angle(image: np.ndarray) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    image_ = image.copy()
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largest_contour = contours[0]
    min_area_rect = cv2.minAreaRect(largest_contour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    image_ = image.copy()
    h, w = image_.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_ = cv2.warpAffine(image_, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image_


def deskew(image: np.ndarray) -> np.ndarray:
    angle = get_skew_angle(image)

    # si da un valor verdura, mejor no hacemos nada
    if abs(angle) > 30:
        return image

    return rotate_image(image, -1.0 * angle)
