from glob import glob
from os.path import join
from pathlib import Path

import click
import matplotlib as mpl
import pandas as pd
from cv2 import cv2

from pytcherplants.utils import row_h, row_s, row_v, row_date, row_treatment, row_title, row_hsv
from pytcherplants.segmentation import segment_plants
from pytcherplants.colors import RESULT_HEADERS, color_analysis, cumulative_color_analysis
import pytcherplants.pixel_classification as pc

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
    input_stem = Path(input).stem
    print(f"Running Ilastik pixel classification on image {input_stem}")
    pc.ilastik_classify(input, output)


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
    print(f"Post-processing Ilastik pixel classification image {input_stem} with mask {mask_stem}")
    pc.postprocess(input, mask, output)


@cli.group()
def colors():
    pass


@colors.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
@click.option('--count', '-c', required=False, type=int, default=1)
@click.option('--min_area', '-m', required=False, type=int)
def analyze(input, output, filetypes, count, min_area):
    if Path(input).is_dir():
        # get files of supported types (currently PNG and JPG)
        if len(filetypes) == 0: filetypes = ['png', 'jpg']
        extensions = [e for es in [[extension.lower(), extension.upper()] for extension in filetypes] for e in es]
        patterns = [join(input, f"*.{p}") for p in extensions]
        files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])

        print(f"Running individual color analysis...")
        plants = {Path(file).stem: segment_plants(file, output, count, min_area) for file in files}
        data = [r for rr in [color_analysis(name, output, pts) for name, pts, in plants.items()] for r in rr]

        print(f"Constructing data frame")
        frame = pd.DataFrame(data, columns=RESULT_HEADERS)

        print(f"Running cumulative color analysis")
        cumulative_color_analysis(frame, output)

    elif Path(input).is_file():
        print(f"Running individual color analysis...")
        data = []
        if count == 1:
            plant = segment_plants(input, output, 1, None)[0]
            plant_name = Path(input).stem
            data = color_analysis(plant, plant_name, output)
            cv2.imwrite(f"{join(output, plant_name + '.png')}", plant.copy())
        else:
            plants = segment_plants(input, output, count, min_area)
            for i, plant in enumerate(plants):
                plant_name = Path(input).stem + '.' + str(i)
                data = data + color_analysis(plant, plant_name, output)
                cv2.imwrite(f"{join(output, plant_name + '.png')}", plant.copy())

        print(f"Constructing data frame")
        frame = pd.DataFrame(data, columns=RESULT_HEADERS)

        print(f"Running cumulative color analysis...")
        cumulative_color_analysis(frame, output)
    else:
        raise ValueError(f"Invalid input path: {input}")


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


# @gpoints.group()
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
# @click.option('--images', '-i', required=True, type=str)
# def train(labels, images):
#     """
#     Trains a convolutional neural network model for density estimation of growth points.
#
#     Adapted from the Deep Plant Phenomics example:
#         https://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Object-Counting-with-Heatmaps/
#
#     :param labels: The label file path (CSV format: image name, x1, y1, x2, y2, ...)
#     :param images: The image directory path
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
#     model.load_heatmap_dataset_with_csv_from_directory(images, Path(labels).name)
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
