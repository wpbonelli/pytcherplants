from os import listdir
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

import pytcherplants.cli as cli

group_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.jpg'
plant_path = 'samples/plants/1_14_19.10_30_20.?.jpg'
masked_path = 'samples/masks/1_14_19.10_30_20.5V4B3121.masked.jpg'


def test_color_analysis_count_0():
    with TemporaryDirectory() as output_path:
        input_stem = Path(masked_path).stem
        runner = CliRunner()
        runner.invoke(cli.color_analysis, ['-i', masked_path, '-o', output_path])

        results = listdir(output_path)
        assert len(results) == 1
        assert f"{input_stem}.colors.csv" in results


def test_color_analysis_count_1():
    with TemporaryDirectory() as output_path:
        input_stem = Path(masked_path).stem
        runner = CliRunner()
        runner.invoke(cli.color_analysis, ['-i', masked_path, '-o', output_path, '-c', 1])

        results = listdir(output_path)
        assert len(results) == 4
        assert f"{input_stem}.plants.png" in results
        assert f"{input_stem}.plants.csv" in results
        assert f"{input_stem}.1.colors.csv" in results
        assert f"{input_stem}.1.png" in results


def test_color_analysis_count_2():
    with TemporaryDirectory() as output_path:
        input_stem = Path(masked_path).stem
        runner = CliRunner()
        runner.invoke(cli.analyze, ['-i', masked_path, '-o', output_path, '-c', 2])

        results = listdir(output_path)
        assert len(results) == 6
        assert f"{input_stem}.plants.png" in results
        assert f"{input_stem}.plants.csv" in results
        assert f"{input_stem}.1.colors.csv" in results
        assert f"{input_stem}.1.png" in results
        assert f"{input_stem}.2.colors.csv" in results
        assert f"{input_stem}.2.png" in results
