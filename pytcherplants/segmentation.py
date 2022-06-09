from heapq import nlargest
from typing import List

import cv2
import numpy as np


def segment_plants(
        image_path: str,
        count: int = None,
        min_area: int = None) -> List[np.ndarray]:
    """
    Segment plants from background pixels.
    Produces a CSV file containing plant dimensions and PNG groups of each cropped out of the original image.

    :param image_path: The path to the image file
    :param count: The number of plants in the image (automatically detected if not provided)
    :param min_area: The minimum plant area
    :return The cropped plant images
    """

    print(f"Looking for {count} plants with minimum area {min_area} in {image_path}")

    print(f"Applying Gaussian blur")
    image = cv2.imread(image_path)
    blurred = cv2.blur(image, (25, 25))
    blurred = cv2.GaussianBlur(blurred, (11, 75), cv2.BORDER_DEFAULT)

    print(f"Applying selective color mask")
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([179, 255, 255]))

    print(f"Dilating image")
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))
    masked = cv2.bitwise_and(image, image, mask=dilated)

    print(f"Detecting contours")
    labelled = masked.copy()
    _, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(labelled, contours, -1, 255, 3)
    div = 8
    keep = count if count is not None and count > 0 else 1
    if keep == 1:
        min_area = 1
    else:
        min_area = min_area if min_area is not None and min_area > 0 else ((image.shape[0] / div) * (image.shape[1] / div))
    largest = [c for c in nlargest(keep, contours, key=cv2.contourArea) if cv2.contourArea(c) > min_area]
    print(f"Found {len(largest)} plants (minimum area: {min_area} pixels)")

    print(f"Cropping contours")
    plants = []
    for i, c in enumerate(largest):
        x, y, w, h = cv2.boundingRect(c)
        plant = masked.copy()
        plants.append(plant[y:y + h, x:x + w])
        cv2.rectangle(labelled, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(labelled, f"plant {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (36, 255, 12), 3)

    return plants, labelled
