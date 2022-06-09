from colorsys import rgb_to_hsv
from glob import glob
from os.path import join
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

from pytcherplants.clustering import get_clusters
from pytcherplants.utils import hex2rgb

HEADERS = ['Image', 'Hex', 'R', 'G', 'B', 'H', 'S', 'V', 'Freq', 'Dens']


def analyze_directory(
        directory_path: str,
        filetypes: List[str] = None) -> pd.DataFrame:
    if filetypes is None or len(filetypes) == 0: filetypes = ['png', 'jpg', 'tiff']  # default to PNG, JPG, and TIFF

    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in filetypes] for e in es]
    patterns = [join(directory_path, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
    rows = []

    for file in files:
        print(f"Running color clustering for image {file}")
        image = cv2.imread(file)
        rows = rows + analyze_image(image=image, name=file)

    return pd.DataFrame(rows, columns=HEADERS)


def analyze_file(image_path: str):
    image_name = Path(image_path).stem
    print(f"Running color clustering for image {image_name}")
    image = cv2.imread(image_path)
    rows = analyze_image(image=image, name=image_name)
    return pd.DataFrame(rows, columns=HEADERS)


def analyze_image(image: np.ndarray, name: str) -> List[List[str]]:
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    clusters = get_clusters(rgb)
    total = sum(clusters.values())
    rows = []

    for hex, freq in clusters.items():
        r, g, b = hex2rgb(hex)
        h, s, v = rgb_to_hsv(r, g, b)
        dens = freq / total
        rows.append([name, hex, r, g, b, h, s, v, freq, dens])

    return rows
