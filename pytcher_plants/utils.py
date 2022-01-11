from colorsys import hsv_to_rgb, rgb_to_hsv

import cv2
import matplotlib as mpl
import numpy as np


def hex2rgb(color):
    """
    Converts the given hexadecimal color code to RGB (on 1-256 scale).

    Referenced from https://stackoverflow.com/a/29643643/6514033
    :param color: the hex code
    :return: the RGB values, in a tuple
    """
    return mpl.colors.to_rgb(color)


def rgb2hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def hue_to_rgb(hue):
    r, g, b = hsv_to_rgb(hue, 0.7, 0.7)
    return float(r), float(g), float(b)


def hue_to_rgb_formatted(k):
    r, g, b = hue_to_rgb(float(k / 360))
    return f"rgb({int(r * 256)},{int(g * 256)},{int(b * 256)})"


def row_to_hsv(row):
    hsv = rgb_to_hsv(float(row['R']), float(row['G']), float(row['B']))
    return [hsv[0], hsv[1], hsv[2]]


def color_analysis(image, i, k=10):
    print(f"K-means color clustering for plant {i}, k = {k}...")

    z = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((image.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    print(f"Plant {i} pre-averaged with k = {k} clusters")

    # remove blues (pot and label), second pass (slightly different range than before)
    lower_blue = np.array([0, 38, 40])
    upper_blue = np.array([39, 255, 255])

    hsv = cv2.cvtColor(reduced, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    masked = cv2.bitwise_and(image, image, mask=mask)

    print(f"Plant {i} blues removed")

    mask = cv2.cvtColor(np.zeros_like(reduced), cv2.COLOR_RGB2GRAY)
    res = np.zeros_like(reduced)
    _, thresh = cv2.threshold(cv2.cvtColor(masked.copy(), cv2.COLOR_BGR2GRAY), 60, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    res[mask == 255] = image[mask == 255]
    print(f"Plant {i} re-masked after secondary preprocessing")

    z = np.float32(res.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((res.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    print(f"Plant {i} averaged with k = {k} clusters:")
    # cv2.imwrite(f"{join(output_directory, base_name + '.reduced.png')}", reduced)

    counts = dict()
    ex_reduced = None
    for ii, cc in enumerate(centers):
        if all(c < 1 for c in cc):
            print(f"Plant {i} color cluster {ii} is background, ignoring")
            continue

        hex = rgb2hex(cc)
        print(f"Plant {i} color cluster {ii}: {hex}")

        mask = cv2.inRange(labels, ii, ii)
        mask = np.dstack([mask] * 3)  # Make it 3 channel
        ex_reduced = cv2.bitwise_and(reduced, mask)
        counts[hex] = len(np.nonzero(ex_reduced)[0])

    return counts, ex_reduced if ex_reduced is not None else np.zeros_like(reduced)


def get_treatment(row):
    image_name = row['Image'].lower()
    if 'control' in image_name: return 'Control'
    elif 'maxsea' in image_name: return 'MaxSea'
    elif 'calmag' in image_name: return 'CalMag'
    elif '10_30_20' in image_name: return 'BloomBoost'
    else: return np.NaN
