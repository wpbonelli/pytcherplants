import argparse
from glob import glob
from os.path import join
from pathlib import Path

import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from heapq import nlargest
from pprint import pprint
import seaborn as sns
import csv

mpl.rcParams['figure.dpi'] = 300


def process(file, output_directory, base_name, count=6):
    image = cv2.imread(file)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply blur
    blur = cv2.blur(rgb, (25, 25))
    gblur = cv2.GaussianBlur(blur, (11, 75), cv2.BORDER_DEFAULT)
    cv2.imwrite(f"{join(output_directory, base_name + '.gblur.png')}", gblur)

    # apply color mask
    hsv = cv2.cvtColor(gblur, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))
    masked = cv2.bitwise_and(image, image, mask=dilated)
    rgb_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{join(output_directory, base_name + '.masked.png')}", rgb_masked)

    # find contours
    ctrs = rgb_masked.copy()
    _, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ctrs, contours, -1, 255, 3)
    keep = 6
    largest = nlargest(keep, contours, key=cv2.contourArea)
    cropped = []

    for i, c in enumerate(largest):
        # crop to bounding rectangle, save, and show cropped image
        x, y, w, h = cv2.boundingRect(c)
        ctr = rgb_masked.copy()
        cropped.append(ctr[y:y + h, x:x + w])
        print(f"Plant {i} cropped")
        cv2.imsave(join(output_directory, base_name + '.' + str(i) + '.cropped.png'), cropped[-1])

        # draw contour on full image
        cv2.rectangle(ctrs, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(ctrs, f"Plant {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (36, 255, 12), 3)

    cv2.imsave(join(output_directory, base_name + '.contours.png'), ctrs)


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
        print(f"Found {len(files)} images")
        for file in files:
            print(f"Processing {file}")
            process(file, output, Path(file).stem, count)
    else:
        print(f"Processing {input}")
        process(input, output, Path(input).stem, count)
