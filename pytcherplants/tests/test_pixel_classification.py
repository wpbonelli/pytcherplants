from tempfile import TemporaryDirectory

import numpy as np
import cv2

import pytcherplants.pixel_classification as pc

input_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.jpg'  # original image
imask_path = 'samples/masks/groups/1_14_19.10_30_20.5V4B3121_segmented.tiff'  # before post-processing
pmask_path = 'samples/masks/groups/1_14_19.10_30_20.5V4B3121_masked.jpg'  # after post-processing


def test_ilastik_classify():
    with TemporaryDirectory() as output_path:
        classified = pc.classify(input_path, output_path)
        imask = cv2.imread(imask_path)
        # for now, just look at the output manually to make sure it's working


def test_postprocess():
    with TemporaryDirectory() as output_path:
        mask, masked = pc.postprocess(input_path, imask_path, output_path)
        assert np.logical_or(mask == 0, mask == 255).all()
        assert np.logical_and(masked != 0, masked != 255).any()
