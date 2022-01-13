import csv
from glob import glob
from heapq import nlargest
from os.path import join
from pathlib import Path
from pprint import pprint

import click
import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd

from pytcher_plants.color import rgb_analysis, hsv_analysis, color_analysis
from pytcher_plants.utils import row_to_hsv, hex2rgb, get_treatment

mpl.rcParams['figure.dpi'] = 300


def analyze_file(file, output_directory, base_name, count: int = None, min_area: int = None):
    """
    Analyze a single image file.

    :param file: The image file
    :param output_directory: The output directory
    :param base_name: The file's basename (without extension)
    :param count: The number of plants in the image (automatically detected if not provided)
    """

    # load image and apply blur
    image = cv2.imread(file)
    blur = cv2.blur(image, (25, 25))
    gblur = cv2.GaussianBlur(blur, (11, 75), cv2.BORDER_DEFAULT)
    # cv2.imwrite(f"{join(output_directory, base_name + '.blurred.png')}", gblur)

    # apply color mask
    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])
    hsv = cv2.cvtColor(gblur, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))
    masked = cv2.bitwise_and(image, image, mask=dilated)
    # cv2.imwrite(f"{join(output_directory, base_name + '.masked.png')}", masked)

    # find and crop to contours
    ctrs = masked.copy()
    _, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ctrs, contours, -1, 255, 3)
    keep = count if count is not None and count > 0 else 100  # what is a suitable maximum?
    divisor = 8
    min_area = min_area if min_area is not None and min_area > 0 else ((image.shape[0] / divisor) * (image.shape[1] / divisor))
    print(f"Keeping top {keep} contours with area > {min_area}...")
    largest = [c for c in nlargest(keep, contours, key=cv2.contourArea) if cv2.contourArea(c) > min_area]
    print(f"Kept {len(largest)} contours")
    cropped = []
    for i, c in enumerate(largest):
        x, y, w, h = cv2.boundingRect(c)
        ctr = masked.copy()
        cropped.append(ctr[y:y + h, x:x + w])
        print(f"Plant {i} cropped")
        cv2.rectangle(ctrs, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(ctrs, f"Plant {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (36, 255, 12), 3)
    cv2.imwrite(join(output_directory, base_name + '.plants.png'), ctrs)

    # color averaging/analysis
    freqs = []
    densities = []
    csv_path = join(output_directory, base_name + '.colors.csv')
    cropped_averaged = []
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Image', 'Plant', 'Hex', 'R', 'G', 'B', 'Freq'])

        for i, crp in enumerate(cropped):
            cpy = crp.copy()
            rgb = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{join(output_directory, base_name + '.' + str(i) + '.png')}", cpy)
            cluster_freqs, averaged = color_analysis(rgb, i)
            cropped_averaged.append(averaged)
            cv2.imwrite(f"{join(output_directory, base_name + '.' + str(i) + '.averaged.png')}", cv2.cvtColor(averaged, cv2.COLOR_RGB2BGR))
            freqs.append(cluster_freqs)
            print(f"Image {base_name} plant {i} color cluster pixel frequencies:")
            pprint(cluster_freqs)
            for h, f in cluster_freqs.items():
                rgb = hex2rgb(h)
                writer.writerow([base_name, str(i), h, rgb[0], rgb[1], rgb[2], f])

            total = sum(cluster_freqs.values())
            cluster_densities = {k: (v / total) for k, v in cluster_freqs.items()}
            densities.append(cluster_densities)
            print(f"Image {base_name} plant {i} color cluster pixel densities:")
            pprint(cluster_densities)


def analyze_results(input_directory, output_directory):
    """
    Post-processing and aggregations, meant to run after images are processed.

    :param input_directory: the directory containing input images
    :param output_directory: the directory containing result files
    :return:
    """

    images = glob(join(input_directory, '*.JPG')) + glob(join(input_directory, '*.jpg'))
    print("Input images:", len(images))

    # list all the result CSVs for each image file (produced by CLI process command)
    results = glob(join(output_directory, '*.CSV')) + glob(join(output_directory, '*.csv'))
    print("Result files:", len(results))

    headers = []
    rows = []
    for result in results:
        with open(result, 'r') as file:
            reader = csv.reader(file)
            if len(headers) == 0:
                headers = next(reader, None)
            else:
                next(reader, None)
            for row in reader: rows.append(row)

    # create dataframe from rows
    df = pd.DataFrame(rows, columns=headers)

    # extract treatment from image name
    df['Treatment'] = df.apply(get_treatment, axis=1)

    # drop rows with unknown treatment (for now)
    df.dropna(how='any', inplace=True)

    # add columns for HSV color representation
    df['H'], df['S'], df['V'] = zip(*df.apply(lambda row: row_to_hsv(row), axis=1))

    # color analysis for each treatment separately
    treatments = list(np.unique(df['Treatment']))
    for treatment in treatments:
        # get subset corresponding to this treatment
        subset = df[df['Treatment'] == treatment]
        print(treatment + ":", len(subset))

        rgb_analysis(subset, treatment, output_directory)
        hsv_analysis(subset, treatment, output_directory)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_directory', '-i', required=True, type=str)
@click.option('--output_directory', '-o', required=True, type=str)
@click.option('--filetypes', '-p', multiple=True, type=str)
@click.option('--count', '-c', required=False, type=int, default=6)
@click.option('--min_area', '-m', required=False, type=int)
def process(input_directory, output_directory, filetypes, count, min_area):

    # by default, support PNGs and JPGs
    if len(filetypes) == 0: filetypes = ['png', 'jpg']
    extensions = filetypes
    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
    patterns = [join(input_directory, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])

    if Path(input_directory).is_dir():
        for file in files:
            print(f"Processing image {file}")
            analyze_file(file, output_directory, Path(file).stem, count, min_area)

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
        print(f"Processing image {input}")
        analyze_file(input, output_directory, Path(input_directory).stem, count, min_area)


@cli.command()
@click.option('--input_directory', '-i', required=True, type=str)
@click.option('--output_directory', '-o', required=True, type=str)
def postprocess(input_directory, output_directory):
    analyze_results(input_directory, output_directory)


if __name__ == '__main__':
    cli()
