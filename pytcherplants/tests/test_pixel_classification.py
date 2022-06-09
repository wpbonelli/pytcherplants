from tempfile import TemporaryDirectory

import pytcherplants.pixel_classification as pc

input_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.jpg'


def test_classify():
    with TemporaryDirectory() as output_path:
        mask, masked = pc.classify(input_path, output_path)
