from colorsys import rgb_to_hsv
from glob import glob
from os.path import join
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

from pytcherplants.clustering import get_clusters
from pytcherplants.segmentation import segment_plants
from pytcherplants.utils import hex2rgb

HEADERS = ['Image', 'Plant', 'Hex', 'R', 'G', 'B', 'H', 'S', 'V', 'Freq', 'Dens']


def analyze_directory(
        directory_path: str,
        output_path: str,
        filetypes: List[str] = None,
        count: int = 0,
        min_area: int = 10) -> pd.DataFrame:
    if count < 0: raise ValueError(f"Plant count must be greater than or equal to 0 (0 disables segmentation)")
    if min_area < 10: raise ValueError(f"Minimum plant area must be greater than 10px")
    if filetypes is None or len(filetypes) == 0: filetypes = ['png', 'jpg', 'tiff']  # default to PNG, JPG, and TIFF

    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in filetypes] for e in es]
    patterns = [join(directory_path, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
    rows = []

    if count > 0:
        images = dict()
        for file in files:
            image_name = Path(file).stem
            print(f"Segmenting plants in image {image_name}...")
            images[image_name] = segment_plants(file, output_path, count, min_area)
            for image_name, plants in images.items():
                for i, plant in enumerate(plants):
                    plant_name = image_name + '.' + str(i + 1)
                    print(f"Running color clustering for plant {plant_name}...")
                    rows = rows + analyze_image(image=plant, image_name=image_name, plant_name=plant_name)
    else:
        images = {Path(file).stem: [cv2.imread(file)] for file in files}
        for image_name, plants in images.items():
            for i, plant in enumerate(plants):
                plant_name = image_name + '.' + str(i + 1)
                print(f"Running color clustering for plant {plant_name}...")
                rows = rows + analyze_image(image=plant, image_name=image_name, plant_name=plant_name)

    return pd.DataFrame(rows, columns=HEADERS)


def analyze_file(image_path: str, output_path: str, count: int, min_area: int):
    if count < 0: raise ValueError(f"Plant count must be greater than or equal to 0 (0 disables segmentation)")
    if min_area < 10: raise ValueError(f"Minimum plant area must be greater than 10px")

    if count > 0:
        print(f"Segmenting plants...")
        images = {Path(image_path).stem: segment_plants(image_path, output_path, count, min_area)}
    else:
        images = {Path(image_path).stem: [cv2.imread(image_path)]}

    rows = []
    for image_name, plants in images.items():
        for i, plant in enumerate(plants):
            plant_name = image_name + '.' + str(i + 1)
            print(f"Running color clustering for plant {plant_name}...")
            rows = rows + analyze_image(image=plant, image_name=image_name, plant_name=plant_name)

    return pd.DataFrame(rows, columns=HEADERS)


def analyze_image(image: np.ndarray, image_name: str, plant_name: str) -> List[List[str]]:
    rows = []
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    clusters, averaged = get_clusters(rgb)
    total = sum(clusters.values())

    for hex, freq in clusters.items():
        r, g, b = hex2rgb(hex)
        h, s, v = rgb_to_hsv(r, g, b)
        dens = freq / total
        rows.append([image_name, plant_name, hex, r, g, b, h, s, v, freq, dens])

    return rows
