from colorsys import rgb_to_hsv
from glob import glob
from os.path import join
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd

from pytcherplants.utils import hex2rgb, rgb2hex

HEADERS = ['Image', 'Hex', 'R', 'G', 'B', 'H', 'S', 'V', 'Freq', 'Prop']


def analyze_directory(directory_path: str, filetypes: List[str] = None, k: int = 10) -> pd.DataFrame:
    if filetypes is None or len(filetypes) == 0: filetypes = ['png', 'jpg', 'tiff']  # default to PNG, JPG, and TIFF

    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in filetypes] for e in es]
    patterns = [join(directory_path, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
    rows = []

    for file in files:
        print(f"Running {k}-means color clustering for image {file}")
        image = cv2.imread(file)
        rows = rows + analyze_image(image=image, name=file, k=k)

    return pd.DataFrame(rows, columns=HEADERS)


def analyze_file(image_path: str, k: int = 10):
    image_name = Path(image_path).stem
    print(f"Running {k}-means color clustering for image {image_name}")
    image = cv2.imread(image_path)
    rows = analyze_image(image=image, name=image_name, k=k)
    return pd.DataFrame(rows, columns=HEADERS)


def analyze_image(image: np.ndarray, name: str, k: int = 10) -> List[List[str]]:
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    clusters = get_clusters(rgb, k=k)
    total = sum(clusters.values())
    rows = []

    for hex, freq in clusters.items():
        r, g, b = hex2rgb(hex)
        h, s, v = rgb_to_hsv(r, g, b)
        dens = freq / total
        rows.append([name, hex, r, g, b, h, s, v, freq, dens])

    return rows


def get_clusters(
        image: np.ndarray,
        k: int = 10,
        filters: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None) -> Dict[str, int]:
    """
    Performs k-means color clustering on the given image.

    :param image: The input image
    :param k: The number of clusters to use
    :param filters: Color bands (in HSV format) to filter out
    :return: A dictionary mapping color hex codes to pixel counts
    """

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
        count = len(np.nonzero(cv2.bitwise_and(filtered, mask))[0])
        counts[hex_code] = count
        print(f"Cluster {ii} ({hex_code}): {count}")

    return counts
