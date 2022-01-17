import csv
from heapq import nlargest
from os.path import join
from pprint import pprint

import cv2
import numpy as np

from pytcher_plants.color import color_averaging
from pytcher_plants.utils import hex2rgb


def preprocess_file(input_file, output_directory, base_name, count: int = None, min_area: int = None):
    """
    Preprocess a single image file. This includes segmentation of plant pots from background and pixel counting/color averaging. A CSV file is produced for each image separately.

    :param input_file: The image file
    :param output_directory: The output directory
    :param base_name: The file's basename (without extension)
    :param count: The number of plants in the image (automatically detected if not provided)
    """

    # load image and apply blur
    print(f"Applying Gaussian blur")
    image = cv2.imread(input_file)
    blur = cv2.blur(image, (25, 25))
    gblur = cv2.GaussianBlur(blur, (11, 75), cv2.BORDER_DEFAULT)
    # cv2.imwrite(f"{join(output_directory, base_name + '.blurred.png')}", gblur)

    # apply color mask
    print(f"Applying color mask")
    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])
    hsv = cv2.cvtColor(gblur, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    print(f"Dilating image")
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))
    masked = cv2.bitwise_and(image, image, mask=dilated)
    # cv2.imwrite(f"{join(output_directory, base_name + '.masked.png')}", masked)

    # find and crop to contours
    print(f"Detecting contours")
    ctrs = masked.copy()
    _, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ctrs, contours, -1, 255, 3)
    keep = count if count is not None and count > 0 else 100  # what is a suitable maximum?
    divisor = 8
    min_area = min_area if min_area is not None and min_area > 0 else ((image.shape[0] / divisor) * (image.shape[1] / divisor))
    largest = [c for c in nlargest(keep, contours, key=cv2.contourArea) if cv2.contourArea(c) > min_area]
    print(f"Found {len(largest)} likely plant contours (minimum area threshold: {min_area} pixels)")
    cropped = []
    for i, c in enumerate(largest):
        x, y, w, h = cv2.boundingRect(c)
        ctr = masked.copy()
        cropped.append(ctr[y:y + h, x:x + w])
        cv2.rectangle(ctrs, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(ctrs, f"plant {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (36, 255, 12), 3)
    cv2.imwrite(join(output_directory, base_name + '.plants.png'), ctrs)

    # color averaging/analysis
    print(f"Pixel-counting and initial color clustering")
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
            clusters, averaged = color_averaging(rgb, i)
            cropped_averaged.append(averaged)
            cv2.imwrite(f"{join(output_directory, base_name + '.' + str(i) + '.averaged.png')}", cv2.cvtColor(averaged, cv2.COLOR_RGB2BGR))
            freqs.append(clusters)
            for hex, freq in clusters.items():
                r, g, b = hex2rgb(hex)
                writer.writerow([base_name, str(i), hex, r, g, b, freq])

            total = sum(clusters.values())
            cluster_densities = {k: (v / total) for k, v in clusters.items()}
            densities.append(cluster_densities)
            print(f"Image {base_name} plant {i} color clusters:")
            pprint(cluster_densities)