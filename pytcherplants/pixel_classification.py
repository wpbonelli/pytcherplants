from os.path import join
from pathlib import Path
import subprocess
from typing import Tuple

import numpy as np
import cv2


def renormalize(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    img[img > 1] = 0
    img[img == 1] = 255
    return img


def classify(input_path: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
    command = f"/opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless " + \
              "--project=/opt/pytcherplants/pytcherplants.ilp " + \
              "--output_format=tiff " + \
              f"--output_filename_format={output_path}/" + "{nickname}.segmented.tiff " + \
              "--export_source='Simple Segmentation'" + \
              f" {input_path}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

    original = cv2.imread(input_path)
    segmented = cv2.imread(join(output_path, f"{Path(input_path).stem}.segmented.tiff"))

    # ilastik pixel segmentation returns [1, 2], with 1=foreground & 2=background
    # we want to convert this to 0=background, 255=foreground
    mask = renormalize(segmented)
    masked = cv2.bitwise_and(original, mask)

    return mask, masked
