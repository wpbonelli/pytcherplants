from pathlib import Path

import pandas as pd
from tempfile import TemporaryDirectory

from cv2 import cv2

from pytcherplants.colors import RESULT_HEADERS, color_analysis, cumulative_color_analysis

pmask_path = 'samples/masks/raw/1_14_19.10_30_20.5V4B3121.masked.jpg'  # after post-processing


def test_color_analysis():
    with TemporaryDirectory() as output_dir:
        image = cv2.imread(pmask_path)
        image_name = Path(pmask_path).stem
        color_analysis(image, image_name, output_dir)


def test_cumulative_color_analysis():
    with TemporaryDirectory() as output_dir:
        df = pd.DataFrame([
            ['1_14_19.10_30_20.5V4B3121.masked', None, '#8e6443', 0.556863, 0.392157, 0.262745, 0.07333337866661226, 0.528169406119638, 0.556863, 52647, 0.044717],
            ['1_14_19.10_30_20.5V4B3121.masked', None, '#6b621e', 0.419608, 0.384314, 0.117647, 0.14718622603581258, 0.7196264132237707, 0.419608, 39366, 0.033436]
        ], columns=RESULT_HEADERS)
        cumulative_color_analysis(df, output_dir)

