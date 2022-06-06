from tempfile import TemporaryDirectory

import numpy as np

from pytcher_plants.ilastik import postprocess_pixel_classification

input_path = 'samples/raw/1_14_19.10_30_20.5V4B3121.JPG'  # original image
imask_path = 'samples/masks/raw/1_14_19.10_30_20.5V4B3121_segmented.tiff'  # before post-processing
pmask_path = 'samples/masks/raw/1_14_19.10_30_20.5V4B3121_masked.jpg'  # after post-processing


def test_postprocess_pixel_classification():
    with TemporaryDirectory() as output_path:
        mask, masked = postprocess_pixel_classification(input_path, imask_path, output_path)
        assert np.logical_or(mask == 0, mask == 255).all()
        assert np.logical_and(masked != 0, masked != 255).any()
