from typing import List, Dict, Tuple

import cv2
import numpy as np

from pytcherplants.utils import rgb2hex


def get_clusters(
        image: np.ndarray,
        k: int = 15,
        filters: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None) -> Dict[str, int]:
    """
    Performs k-means color clustering on the given image.

    :param image: The input image
    :param k: The number of clusters to use
    :param filters: Color bands (in HSV format) to filter out
    :return: A dictionary mapping color hex codes to pixel counts
    """

    print(f"K-means color clustering...")
    z = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((image.shape[:-1]))
    averaged = np.uint8(centers)[labels]

    if filters is not None and len(filters) > 0:
        for band in filters:
            lo_hsv = band[0]
            hi_hsv = band[1]
            print(f"Removing unwanted hue range [HSV ({str(lo_hsv[0])}, {str(lo_hsv[1])}, {str(lo_hsv[2])}) - ({hi_hsv[0]}, {hi_hsv[1]}, {hi_hsv[2]})]")
            lower = np.array(lo_hsv)
            upper = np.array(hi_hsv)
            hsv = cv2.cvtColor(averaged, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            filtered = cv2.bitwise_and(image, image, mask=mask)
    else:
        filtered = image

    counts = dict()
    for ii, cc in enumerate(centers):
        # ignore black or white background
        if all(c < 1 for c in cc):
            print(f"Cluster {ii} is black background, ignoring")
            continue

        if all(c > 254 for c in cc):
            print(f"Cluster {ii} is white background, ignoring")
            continue

        hex_code = rgb2hex(cc)
        mask = np.dstack([cv2.inRange(labels, ii, ii)] * 3)
        counts[hex_code] = len(np.nonzero(cv2.bitwise_and(filtered, mask))[0])
        print(f"Cluster {ii}: {hex_code}")

    return counts


