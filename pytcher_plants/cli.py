from glob import glob
from os.path import join
from pathlib import Path

import click
import matplotlib as mpl
import deepplantphenomics as dpp

from pytcher_plants.color import color_analysis
from pytcher_plants.growth_points import detect_growth_point_labels, growth_point_labels_to_csv_format
from pytcher_plants.preprocess import preprocess_file
from pytcher_plants.utils import hex_to_hue_range

mpl.rcParams['figure.dpi'] = 300


@click.group()
def cli():
    pass


@cli.group()
def colors():
    pass


@cli.group()
def gpoints():
    pass


@cli.group()
def pitchers():
    pass


@cli.command()
@click.option('--input', '-i', required=True, type=str)
@click.option('--output', '-o', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
@click.option('--count', '-c', required=False, type=int, default=6)
@click.option('--min_area', '-m', required=False, type=int)
def preprocess(input, output, filetypes, count, min_area):

    # by default, support PNGs and JPGs
    if len(filetypes) == 0: filetypes = ['png', 'jpg']
    extensions = filetypes
    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
    patterns = [join(input, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])

    if Path(input).is_dir():
        for file in files:
            print(f"Preprocessing image {file}")
            preprocess_file(file, output, Path(file).stem, count, min_area)

        # TODO: kmeans function doesn't seem to be thread-safe
        #  (running it on multiple cores in parallel causes all but 1 to fail)
        #  ...reinstate if workaround found
        #
        # processes = psutil.cpu_count(logical=False)
        # print(f"Using {processes} processes for {len(files)} images")
        # with closing(multiprocessing.Pool(processes=processes)) as pool:
        #     args = [(file, output, Path(file).stem, count) for file in files]
        #     pool.starmap(process, args)
        #     pool.terminate()
    else:
        print(f"Preprocessing image {input}")
        preprocess_file(input, output, Path(input).stem, count, min_area)


@gpoints.command()
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


@gpoints.command()
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


@colors.command()
@click.option('--input_directory', '-i', required=True, type=str)
@click.option('--output_directory', '-o', required=True, type=str)
def analyze(input_directory, output_directory):
    color_analysis(input_directory, output_directory)


if __name__ == '__main__':
    cli()
