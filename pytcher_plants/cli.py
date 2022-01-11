from glob import glob
from os.path import join
from pathlib import Path

import cv2
import click
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from heapq import nlargest
from pprint import pprint
import seaborn as sns
import csv

from pytcher_plants.color import rgb_analysis, hsv_analysis
from pytcher_plants.utils import row_to_hsv, color_analysis, hex2rgb, get_treatment

mpl.rcParams['figure.dpi'] = 300


def analyze_file(file, output_directory, base_name, count: int = None, min_area: int = None):
    """
    Analyze a single image file.

    :param file: The image file
    :param output_directory: The output directory
    :param base_name: The file's basename (without extension)
    :param count: The number of plants in the image (automatically detected if not provided)
    """

    image = cv2.imread(file)
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply blur
    blur = cv2.blur(image, (25, 25))
    gblur = cv2.GaussianBlur(blur, (11, 75), cv2.BORDER_DEFAULT)
    # cv2.imwrite(f"{join(output_directory, base_name + '.blurred.png')}", gblur)
    # sys.exit(1)

    # apply color mask
    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])
    hsv = cv2.cvtColor(gblur, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))
    masked = cv2.bitwise_and(image, image, mask=dilated)
    # cv2.imwrite(f"{join(output_directory, base_name + '.masked.png')}", masked)
    # sys.exit(1)

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
    cv2.imwrite(join(output_directory, base_name + '.contours.png'), ctrs)

    # color analysis
    freqs = []
    densities = []
    csv_path = join(output_directory, base_name + '.colors.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Image', 'Plant', 'Hex', 'R', 'G', 'B', 'Freq'])

        for i, crp in enumerate(cropped):
            cluster_freqs, ex_reduced = color_analysis(cv2.cvtColor(crp.copy(), cv2.COLOR_BGR2RGB), i)
            # cv2.imwrite(f"{join(output_directory, base_name + '.exreduced.png')}", ex_reduced)
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

    for i, f in enumerate(freqs):
        x = list(f.keys())
        y = f.values()
        hp = sns.histplot(x=x, weights=y, hue=x, palette=x, discrete=True)
        plt.xticks(rotation=60)
        plt.legend().remove()
        plt.title(f"Image {base_name} plant {i} color cluster pixel frequencies")
        plt.savefig(join(output_directory, base_name + 'plant' + str(i) + '.avg.freq.png'))
        plt.clf()

    for i, plant in enumerate(freqs):
        freqs_rgb = [(hex2rgb(k), v) for k, v in plant.items()]

        xs = []
        ys = []
        zs = []
        ss = []
        total = sum([f[1] for f in freqs_rgb])

        for f in freqs_rgb:
            xs.append(f[0][0])
            ys.append(f[0][1])
            zs.append(f[0][2])
            ss.append((f[1] / total) * 10000)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, s=ss, c=list(zip(xs, ys, zs)))
        plt.xlabel("R")
        plt.ylabel("G")
        ax.set_zlabel("B")
        plt.savefig(join(output_directory, base_name + 'plant' + str(i) + '.avg.rgb.png'))
        plt.clf()


def analyze_results(input_directory, output_directory):
    """
    Post-processing and aggregations, meant to run after images are processed.

    :param input_directory: the directory containing input images
    :param output_directory: the directory containing result files
    :return:
    """

    images = glob(join(input_directory, '*.JPG')) + glob(join(input_directory, '*.jpg'))
    print("Input images:", len(images))

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

        # hue analysis
        rgb_analysis(treatment, output_directory, subset)
        hsv_analysis(treatment, output_directory, subset)


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
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True, help="Input path")
    # ap.add_argument("-o", "--output", required=True, help="Output path")
    # ap.add_argument("-ft", "--filetypes", required=False, default='png,jpg', help="Image filetypes")
    # ap.add_argument("-c", "--count", required=False, default=6, help="Number of individuals")

    # args = vars(ap.parse_args())
    # input_directory = args['input']
    # output_directory = args['output']
    # extensions = args['filetypes'].split(',') if 'filetypes' in args else []
    # count = int(args['count'])

    if len(filetypes) == 0: filetypes = ['png', 'jpg']
    extensions = filetypes
    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
    patterns = [join(input_directory, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])

    if Path(input_directory).is_dir():
        for file in files:
            print(f"Processing image {file}")
            analyze_file(file, output_directory, Path(file).stem, count, min_area)

        # TODO: kmeans function doesn't seem to be threadsafe
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
