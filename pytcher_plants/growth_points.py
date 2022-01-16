from typing import Tuple, List

import numpy as np
import cv2


def detect_growth_point_labels(image, hue_range: Tuple[int, int]) -> List[Tuple[int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([hue_range[0], 50, 50]), np.array([hue_range[1], 255, 255]))
    target = cv2.bitwise_and(image, image, mask=mask)

    # debugging
    # cv2.imshow('image', target)
    # cv2.waitKey()

    ctrs = target.copy()
    _, thresh = cv2.threshold(cv2.cvtColor(ctrs, cv2.COLOR_BGR2GRAY), 140, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(c) for c in contours]
    moments = [m for m in moments if 'm00' in m and int(m['m00']) != 0]
    centers = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) for m in moments]
    print(f"Found {len(contours)} growth point labels")

    # debugging
    # cv2.drawContours(ctrs, contours, -1, 255, 3)
    # for center in centers: cv2.circle(ctrs, (center[0], center[1]), 10, (0, 255, 0), -1)
    # cv2.imshow('image', ctrs)
    # cv2.waitKey()

    return centers


def growth_point_labels_to_csv_format(labels: List[Tuple[int, int]]):
    return [ll for l in zip([str(label[0]) for label in labels], [str(label[1]) for label in labels]) for ll in l]  # x1, y1, x2, y2, ...
