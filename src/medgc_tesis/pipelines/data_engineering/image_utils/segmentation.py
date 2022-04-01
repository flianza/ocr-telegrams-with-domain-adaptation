import functools
from typing import Dict, Iterable

import cv2
import numpy as np

from medgc_tesis.utils.iterable import window


def encuadrar(matrix: np.ndarray, val=255) -> np.ndarray:
    """
    Hace una metriz cuadrada, centrandola.
    """
    a, b = matrix.shape
    if a > b:
        pad = (a - b) // 2
        offset = 0
        if (a - b) % 2 != 0:
            offset = 1
        padding = ((0, 0), (pad + offset, pad))
    else:
        pad = (b - a) // 2
        offset = 0
        if (b - a) % 2 != 0:
            offset = 1
        padding = ((pad + offset, pad), (0, 0))
    return np.pad(matrix, padding, mode="constant", constant_values=val)


def segmentar_digitos(bloque_digitos: np.ndarray) -> Iterable[np.ndarray]:
    image = bloque_digitos.copy()
    gray = cv2.cvtColor(bloque_digitos, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring and thresholding
    # to reveal the characters on the license plate
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15
    )

    # Perform connected components analysis on the thresholded images and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 70  # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 5  # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for label in np.unique(labels):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the bounding boxes from left to right
    def compare(rect1, rect2):
        return rect1[0] - rect2[0]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    for x, y, w, h in boundingBoxes:
        img_digit = gray[y : (y + h), x : (x + w)]
        yield img_digit


def extraer_votos(
    imagen: np.ndarray,
    cortes_horizontales: Iterable[int],
    cortes_verticales: Iterable[int],
) -> Iterable[Dict[str, np.ndarray]]:
    votos = []

    for y1, y2 in window(cortes_verticales, n=2):
        votos_diputados = []
        votos_senadores = []

        for idx, (x1, x2) in enumerate(window(cortes_horizontales, n=2)):
            # salteamos el primer indice de cada row, porque es el titulo del partido politico
            if idx == 0:
                continue

            # eliminamos un par de pixeles para que no se vean los recuadros negros
            offset = 5
            bloque_digitos = imagen[
                y1 + offset : y2 - offset, x1 + offset : x2 - offset
            ]

            for digito in segmentar_digitos(bloque_digitos):
                digito = encuadrar(digito)
                if idx == 1:
                    votos_senadores.append(digito)
                if idx == 2:
                    votos_diputados.append(digito)

        votos.append({"diputados": votos_diputados, "senadores": votos_senadores})

    return votos
