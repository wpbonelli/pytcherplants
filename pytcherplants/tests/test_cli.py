from os import listdir
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

import pytcherplants.cli as cli

group_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.jpg'
group_masked_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.masked.jpg'
plant_path = 'samples/plants/1_14_19.10_30_20.p001.jpg'
plant_masked_path = 'samples/plants/1_14_19.10_30_20.p001.masked.jpg'


def test_classify():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        runner.invoke(cli.classify, ['-i', plant_path, '-o', output_path])

        results = listdir(output_path)
        assert len(results) == 3

        stem = Path(plant_path).stem
        assert f"{stem}.mask.jpg" in results
        assert f"{stem}.masked.jpg" in results
        assert f"{stem}.segmented.tiff" in results


def test_segment_count_1():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        runner.invoke(cli.segment, ['-i', plant_masked_path, '-o', output_path])

        results = listdir(output_path)
        assert len(results) == 2

        stem = Path(plant_masked_path).stem
        assert f"{stem}.plants.png" in results
        assert f"{stem}.plant.1.png" in results


def test_segment_count_2():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        runner.invoke(cli.segment, ['-i', plant_masked_path, '-o', output_path, '-c', 2])

        results = listdir(output_path)
        assert len(results) == 3

        stem = Path(plant_masked_path).stem
        assert f"{stem}.plants.png" in results
        assert f"{stem}.plant.1.png" in results
        assert f"{stem}.plant.2.png" in results


def test_analyze():
    with TemporaryDirectory() as output_path:
        runner = CliRunner()
        result = runner.invoke(cli.analyze, ['-i', plant_masked_path, '-o', output_path])
        print(result.output)

        results = listdir(output_path)
        assert len(results) == 1
        assert f"{Path(plant_masked_path).stem}.colors.csv" in results
