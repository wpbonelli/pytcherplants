import argparse
import multiprocessing
from contextlib import closing
from glob import glob
from os.path import join
from pathlib import Path

import cv2
import matplotlib as mpl
import psutil
from matplotlib import pyplot as plt
import numpy as np
from heapq import nlargest
from pprint import pprint
import seaborn as sns
import csv

mpl.rcParams['figure.dpi'] = 300


def rgb2hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# referenced from https://stackoverflow.com/a/29643643/6514033
def hex2rgb(color):
    return mpl.colors.to_rgb(color)


def color_analysis(image, output_directory, base_name, i, k=10):
    print(f"K-means color clustering for image {base_name} plant {i}, k = {k}...")

    z = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((image.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    print(f"Image {base_name} plant {i} averaged with k = {k} color clusters:")
    cv2.imwrite(f"{join(output_directory, base_name + '.plant' + str(i) + '.clusters.png')}", cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB))

    counts = dict()

    for ii, cc in enumerate(centers):
        if all(c < 1 for c in cc):
            print(f"Image {base_name} plant {i} color cluster {ii} is background, ignoring")
            continue

        hex = rgb2hex(cc)
        print(f"Image {base_name} plant {i} color cluster {ii}: {hex}")

        mask = cv2.inRange(labels, ii, ii)
        mask = np.dstack([mask] * 3)  # Make it 3 channel
        ex_reduced = cv2.bitwise_and(reduced, mask)
        counts[hex] = len(np.nonzero(ex_reduced)[0])
        cv2.imwrite(f"{join(output_directory, base_name + '.plant' + str(i) + '.cluster' + str(ii) + '.png')}", cv2.cvtColor(ex_reduced, cv2.COLOR_BGR2RGB))

    return counts


def process(file, output_directory, base_name, count=6):
    image = cv2.imread(file)
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply blur
    blur = cv2.blur(image, (65, 45))
    gblur = cv2.GaussianBlur(blur, (27, 195), cv2.BORDER_DEFAULT)
    cv2.imwrite(f"{join(output_directory, base_name + '.blurred.png')}", gblur)

    # apply color mask
    hsv = cv2.cvtColor(gblur, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((7, 7)), 5)
    masked = cv2.bitwise_and(image, image, mask=dilated)
    rgb_masked = masked
    cv2.imwrite(f"{join(output_directory, base_name + '.masked.png')}", rgb_masked)

    # blur again & dilate
    reblur = cv2.blur(rgb_masked, (65, 45))
    dilated = cv2.dilate(reblur, np.ones((95, 95), np.uint8), 15)
    cv2.imwrite(f"{join(output_directory, base_name + '.dilated.png')}", dilated)

    # find contours
    ctrs = rgb_masked.copy()
    # _, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = nlargest(count, contours, key=cv2.contourArea)
    cropped = []

    for i, c in enumerate(largest):
        # crop to bounding rectangle, save, and show cropped image
        x, y, w, h = cv2.boundingRect(c)
        ctr = rgb_masked.copy()
        cropped.append(ctr[y:y + h, x:x + w])
        print(f"Image {base_name} plant {i} cropped")
        cv2.imwrite(join(output_directory, base_name + '.plant' + str(i) + '.cropped.png'), cropped[-1])

        # draw contour on full image
        cv2.rectangle(ctrs, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(ctrs, f"plant {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (36, 255, 12), 3)

    cv2.imwrite(join(output_directory, base_name + '.contours.png'), ctrs)

    # color analysis
    freqs = []
    densities = []

    csv_path = join(output_directory, base_name + '.colors.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Image', 'Plant', 'Hex', 'R', 'G', 'B', 'Freq'])

        for i, crp in enumerate(cropped):
            cluster_freqs = color_analysis(cv2.cvtColor(crp.copy(), cv2.COLOR_BGR2RGB), output_directory, base_name, i)
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

        for f in freqs_rgb:
            xs.append(f[0][0])
            ys.append(f[0][1])
            zs.append(f[0][2])
            ss.append(f[1] / 100)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, s=ss, c=list(zip(xs, ys, zs)))
        plt.xlabel("R")
        plt.ylabel("G")
        ax.set_zlabel("B")
        plt.savefig(join(output_directory, base_name + 'plant' + str(i) + '.avg.rgb.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input path")
    ap.add_argument("-o", "--output", required=True, help="Output path")
    ap.add_argument("-ft", "--filetypes", required=False, default='png,jpg', help="Image filetypes")
    ap.add_argument("-c", "--count", required=False, default=6, help="Number of individuals")

    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    extensions = args['filetypes'].split(',') if 'filetypes' in args else []
    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
    patterns = [join(input, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
    count = int(args['count'])

    if Path(input).is_dir():
        processes = psutil.cpu_count(logical=False)
        print(f"Using {processes} processes for {len(files)} images")
        with closing(multiprocessing.Pool(processes=processes)) as pool:
            args = [(file, output, Path(file).stem, count) for file in files]
            pool.starmap(process, args)
            pool.terminate()
        # for file in files:
        #     print(f"Processing image {file}")
        #     process(file, output, Path(file).stem, count)
    else:
        print(f"Processing image {input}")
        process(input, output, Path(input).stem, count)
