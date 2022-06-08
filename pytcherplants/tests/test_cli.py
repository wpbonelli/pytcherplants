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
        runner.invoke(cli.analyze, ['-i', pmask_path, '-o', output_path])

        results = listdir(output_path)
        assert len(results) == 1
        assert '1_14_19.10_30_20.5V4B3121.masked.colors.csv' in results


def test_colors_analyze_count_1():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        runner.invoke(cli.analyze, ['-i', pmask_path, '-o', output_path, '-c', 1])

        results = listdir(output_path)
        assert len(results) == 4
        assert '1_14_19.10_30_20.5V4B3121.masked.plants.png' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.plants.csv' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.1.colors.csv' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.1.png' in results


def test_colors_analyze_count_2():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        runner.invoke(cli.analyze, ['-i', pmask_path, '-o', output_path, '-c', 2])

        results = listdir(output_path)
        assert len(results) == 6
        assert '1_14_19.10_30_20.5V4B3121.masked.plants.png' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.plants.csv' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.1.colors.csv' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.1.png' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.2.colors.csv' in results
        assert '1_14_19.10_30_20.5V4B3121.masked.2.png' in results
