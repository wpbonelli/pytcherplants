import csv
from glob import glob
from os.path import join
from pathlib import Path

import click
import deepplantphenomics as dpp
import matplotlib as mpl
import pandas as pd
from cv2 import cv2

from pytcherplants.gpoints import detect_growth_point_labels, growth_point_labels_to_csv_format
from pytcherplants.colors import TRAITS_HEADERS, get_pots, get_pot_traits, cumulative_color_analysis
from pytcherplants.utils import hex_to_hue_range
from pytcherplants.ilastik import postprocess_pixel_classification

mpl.rcParams['figure.dpi'] = 300


@click.group()
def cli():
    pass


@cli.group()
def ilastik():
    pass


@ilastik.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--mask', '-m', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
def postpc(input, mask, output):
    if not Path(input).is_file(): raise ValueError(f"Input must be a valid file path")
    if not Path(mask).is_file(): raise ValueError(f"Mask must be a valid file path")
    if not Path(output).is_dir(): raise ValueError(f"Output must be a valid directory path")

    input_stem = Path(input).stem
    mask_stem = Path(mask).stem
    print(f"Post-processing Ilastik pixel classification image {input_stem} with mask {mask_stem}")
    postprocess_pixel_classification(input, mask, output)


@cli.group()
def gpoints():
    pass


@gpoints.group()
def hull():
    pass


@gpoints.group()
def skel():
    pass


@gpoints.group()
def cnn():
    pass


@cnn.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
@click.option('--color', '-c', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
def load_labels(input, output, color, filetypes):
    color = color if str(color).startswith('#') else f"#{color}"
    hue_lo, hue_hi = hex_to_hue_range(color)
    rows = []
    if Path(input).is_dir():
        if len(filetypes) == 0: filetypes = ['png', 'jpg']
        extensions = filetypes
        extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
        patterns = [join(input, f"*.{p}") for p in extensions]
        files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])

        for file in files:
            stem = Path(file).stem
            print(f"Detecting growth point labels with color {color} (hue range {hue_lo}-{hue_hi}) in image {stem}")
            image = cv2.imread(file)
            labels = detect_growth_point_labels(image, (hue_lo, hue_hi))
            rows.append([stem] + growth_point_labels_to_csv_format(labels))
    else:
        stem = Path(input).stem
        print(f"Detecting growth point labels in image {stem}")
        image = cv2.imread(input)
        labels = detect_growth_point_labels(image, (hue_lo, hue_hi))
        rows.append([stem] + growth_point_labels_to_csv_format(labels))

    csv_path = join(output, 'labels.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows: writer.writerow(row)


@cnn.command()
@click.option('--labels', '-l', required=True, type=str)
@click.option('--images', '-i', required=True, type=str)
def train(labels, images):
    """
    Trains a convolutional neural network model for density estimation of growth points.

    Adapted from the Deep Plant Phenomics example:
        https://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Object-Counting-with-Heatmaps/

    :param labels: The label file path (CSV format: image name, x1, y1, x2, y2, ...)
    :param images: The image directory path
    """

    channels = 3  # RGB
    model = dpp.HeatmapObjectCountingModel(debug=True, load_from_saved=False)
    model.set_image_dimensions(128, 128, channels)
    model.set_batch_size(32)
    model.set_learning_rate(0.0001)
    model.set_maximum_training_epochs(25)
    model.set_test_split(0.75)
    model.set_validation_split(0.0)

    # load dataset
    model.set_density_map_sigma(4.0)
    model.load_heatmap_dataset_with_csv_from_directory(images, Path(labels).name)

    # define architecture
    model.add_input_layer()
    model.add_convolutional_layer(filter_dimension=[3, 3, 3, 16], stride_length=1, activation_function='relu')
    model.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
    model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')
    model.add_output_layer()

    # begin
    model.begin_training()


@cli.group()
def colors():
    pass


@colors.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
@click.option('--count', '-c', required=False, type=int, default=6)
@click.option('--min_area', '-m', required=False, type=int)
def analyze(input, output, filetypes, count, min_area):
    if Path(input).is_dir():
        # by default, support PNGs and JPGs
        if len(filetypes) == 0: filetypes = ['png', 'jpg']
        extensions = [e for es in [[extension.lower(), extension.upper()] for extension in filetypes] for e in es]
        patterns = [join(input, f"*.{p}") for p in extensions]
        files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
        pots_by_image = {Path(file).stem: get_pots(file, output, count, min_area) for file in files}
        rows = [get_pot_traits(image_name, output, image_pots) for image_name, image_pots in pots_by_image.items()]
        rows = [r for rr in rows for r in rr]  # flatten 2d list to 1d
        cumulative_color_analysis(pd.DataFrame(rows, columns=TRAITS_HEADERS), output)
    elif Path(input).is_file():
        image_pots = get_pots(input, output, count, min_area)
        rows = get_pot_traits(Path(input).stem, output, image_pots)
        cumulative_color_analysis(pd.DataFrame(rows, columns=TRAITS_HEADERS), output)
    else:
        raise ValueError(f"Invalid input path: {input}")


if __name__ == '__main__':
    cli()
