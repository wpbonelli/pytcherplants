import uuid
import unittest
from tempfile import TemporaryDirectory
from os import listdir

from click.testing import CliRunner
import pytcherplants.cli as cli

input_path = 'samples/raw/1_14_19.10_30_20.5V4B3121.JPG'  # original image
imask_path = 'samples/masks/raw/1_14_19.10_30_20.5V4B3121_segmented.tiff'  # before post-processing
pmask_path = 'samples/masks/raw/1_14_19.10_30_20.5V4B3121.masked.jpg'  # after post-processing


def test_colors_analyze_count_0():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        result = runner.invoke(cli.analyze, ['-i', pmask_path, '-o', output_path])

        results = listdir(output_path)
        print(results)


def test_colors_analyze_count_1():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        result = runner.invoke(cli.analyze, ['-i', pmask_path, '-o', output_path])

        results = listdir(output_path)
        print(results)
