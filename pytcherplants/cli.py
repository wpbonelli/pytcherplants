from colorsys import rgb_to_hsv
from os.path import join
from pathlib import Path

import click
import cv2
import matplotlib as mpl
import pandas as pd

import pytcherplants.color_analysis as ca
import pytcherplants.pixel_classification as pc
from pytcherplants.clustering import get_clusters
from pytcherplants.color_analysis import HEADERS
from pytcherplants.segmentation import segment_plants
from pytcherplants.utils import hex2rgb

mpl.rcParams['figure.dpi'] = 300


@click.group()
def cli():
    pass


@cli.group()
def pixel_classification():
    pass


@pixel_classification.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
def classify(input, output):
    if not Path(input).is_file(): raise ValueError(f"Input must be a valid file path")
    if not Path(output).is_dir(): raise ValueError(f"Output must be a valid directory path")

    input_stem = Path(input).stem
    print(f"Running Ilastik pixel classification on image {input_stem}")
    pc.classify(input, output)


@pixel_classification.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--mask', '-m', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
def postprocess(input, mask, output):
    if not Path(input).is_file(): raise ValueError(f"Input must be a valid file path")
    if not Path(mask).is_file(): raise ValueError(f"Mask must be a valid file path")
    if not Path(output).is_dir(): raise ValueError(f"Output must be a valid directory path")

    input_stem = Path(input).stem
    mask_stem = Path(mask).stem
    print(f"Post-processing Ilastik pixel classification for image {input_stem} with mask {mask_stem}")
    pc.postprocess(input, mask, output)


@cli.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
@click.option('--count', '-c', required=False, type=int, default=0)
@click.option('--min_area', '-m', required=False, type=int)
def color_analysis(input, output, filetypes, count, min_area):
    output_path = Path(output)
    if not output_path.is_dir(): raise ValueError(f"Output must be a valid directory path")

    input_path = Path(input)
    input_name = input_path.stem
    if input_path.is_dir():
        print(f"Performing color analysis for directory {input_name}")
        df = ca.analyze_directory(input, output, filetypes, count, min_area)
        df.to_csv(join(output, f"{input_name}.colors.csv"))
    elif input_path.is_file():
        print(f"Performing color analysis for image {input_name}")
        df = ca.analyze_file(input, output, count, min_area)
        df.to_csv(join(output, f"{input_name}.colors.csv"))
    else:
        raise ValueError(f"Invalid input path: {input_path}")


# @cli.group()
# def growth_points():
#     pass
#
#
# @growth_points.group()
# def hull():
#     pass
#
#
# @growth_points.group()
# def skel():
#     pass


# @growth_points.group()
# def cnn():
#     pass


# @cnn.command()
# @click.option('--input', '-i', required=True, type=str)
# @click.option('--output', '-o', required=True, type=str)
# @click.option('--color', '-c', required=True, type=str)
# @click.option('--filetypes', '-p', multiple=True, type=str)
# def load_labels(input, output, color, filetypes):
#     color = color if str(color).startswith('#') else f"#{color}"
#     hue_lo, hue_hi = hex_to_hue_range(color)
#     rows = []
#     if Path(input).is_dir():
#         if len(filetypes) == 0: filetypes = ['png', 'jpg']
#         extensions = filetypes
#         extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
#         patterns = [join(input, f"*.{p}") for p in extensions]
#         files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
#
#         for file in files:
#             stem = Path(file).stem
#             print(f"Detecting growth point labels with color {color} (hue range {hue_lo}-{hue_hi}) in image {stem}")
#             image = cv2.imread(file)
#             labels = detect_growth_point_labels(image, (hue_lo, hue_hi))
#             rows.append([stem] + growth_point_labels_to_csv_format(labels))
#     else:
#         stem = Path(input).stem
#         print(f"Detecting growth point labels in image {stem}")
#         image = cv2.imread(input)
#         labels = detect_growth_point_labels(image, (hue_lo, hue_hi))
#         rows.append([stem] + growth_point_labels_to_csv_format(labels))
#
#     csv_path = join(output, 'labels.csv')
#     with open(csv_path, 'w') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         for row in rows: writer.writerow(row)


# @cnn.command()
# @click.option('--labels', '-l', required=True, type=str)
# @click.option('--groups', '-i', required=True, type=str)
# def train(labels, groups):
#     """
#     Trains a convolutional neural network model for density estimation of growth points.
#
#     Adapted from the Deep Plant Phenomics example:
#         https://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Object-Counting-with-Heatmaps/
#
#     :param labels: The label file path (CSV format: image name, x1, y1, x2, y2, ...)
#     :param groups: The image directory path
#     """
#
#     channels = 3  # RGB
#     model = dpp.HeatmapObjectCountingModel(debug=True, load_from_saved=False)
#     model.set_image_dimensions(128, 128, channels)
#     model.set_batch_size(32)
#     model.set_learning_rate(0.0001)
#     model.set_maximum_training_epochs(25)
#     model.set_test_split(0.75)
#     model.set_validation_split(0.0)
#
#     # load dataset
#     model.set_density_map_sigma(4.0)
#     model.load_heatmap_dataset_with_csv_from_directory(groups, Path(labels).name)
#
#     # define architecture
#     model.add_input_layer()
#     model.add_convolutional_layer(filter_dimension=[3, 3, 3, 16], stride_length=1, activation_function='relu')
#     model.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
#     model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')
#     model.add_output_layer()
#
#     # begin
#     model.begin_training()


if __name__ == '__main__':
    cli()
