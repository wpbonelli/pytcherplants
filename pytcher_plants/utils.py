from colorsys import hsv_to_rgb, rgb_to_hsv
from typing import Tuple

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


def hex_to_hue_range(color: str, radius: int = 3) -> Tuple[int, int]:
    """
    Converts a hexadecimal color code to a hue range (interval of +/- radius around corresponding hue)

    :param color: The hex color code
    :param radius: The amount to pad the interval around the hue
    :return: The hue range
    """

    r, g, b = hex2rgb(color)
    h, _, _ = rgb_to_hsv(r, g, b)
    h = int(h * 179)
    return h - radius, h + radius


def hue_to_rgb(hue):
    r, g, b = hsv_to_rgb(hue, 0.7, 0.7)
    return float(r), float(g), float(b)


def hue_to_rgb_formatted(k):
    r, g, b = hue_to_rgb(float(k / 360))
    return f"rgb({int(r * 256)},{int(g * 256)},{int(b * 256)})"


def row_to_hsv(row):
    hsv = rgb_to_hsv(float(row['R']), float(row['G']), float(row['B']))
    return [hsv[0], hsv[1], hsv[2]]


def get_treatment(row):
    image_name = row['Image'].lower()
    if 'control' in image_name: return 'Control'
    elif 'maxsea' in image_name: return 'MaxSea'
    elif 'calmag' in image_name: return 'CalMag'
    elif '10_30_20' in image_name: return 'BloomBoost'
    else: return np.NaN
