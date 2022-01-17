from collections import Counter, OrderedDict
from os.path import join
from pprint import pprint
from typing import Tuple

import cv2
import numpy as np

import pandas as pd
import seaborn as sns
from scipy.cluster.vq import kmeans2
from plotly import express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pytcher_plants.utils import hue_to_rgb_formatted, rgb2hex


def rgb_analysis(data: pd.DataFrame, treatment: str, output_directory: str = '.'):
    subset_rgb = data[['R', 'G', 'B']].astype(float).values.tolist()
    k = 25
    centers, labels = kmeans2(subset_rgb, k)
    counter = dict(Counter(labels))
    counts = {(abs(int(float(c[0]) * 256)), abs(int(float(c[1]) * 256)), abs(int(float(c[2]) * 256))): counter[l] for c, l in zip(centers, labels)}
    total = sum(counts.values())
    props = {k: (v / total) for k, v in counts.items()}

    x = list([rgb2hex(k).replace('-', '') for k in props.keys()])
    y = list(props.values())
    sns.histplot(x=x, weights=y, hue=x, palette=x, discrete=True)
    plt.xticks(rotation=60)
    plt.legend().remove()
    plt.title(f"{treatment} color distribution")
    plt.savefig(join(output_directory, f"{treatment}.k{k}.dist.png"))
    # plt.show()
    plt.clf()

    fig = go.Figure()
    r = [k[0] for k in props.keys()]
    g = [k[1] for k in props.keys()]
    b = [k[2] for k in props.keys()]
    colors_map = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in props.keys()]
    sizes_map = list([v * 1000 for v in props.values()])
    trace = dict(type='scatter3d', x=r, y=g, z=b, mode='markers', marker=dict(color=colors_map, size=sizes_map))
    fig.add_trace(trace)
    fig.update_layout(title=treatment, scene=dict(xaxis_title='G', yaxis_title='R', zaxis_title='B'))
    # fig.show()
    fig.write_image(join(output_directory, treatment + '.rgb.3d.png'))


def hsv_analysis(data: pd.DataFrame, treatment: str, output_directory: str):
    # (72 equally spaced from 5 to 355)
    divisor = 5
    ranges = [((k * divisor) + divisor) for k in range(0, int(360 / divisor))]

    # format HSV columns, convert to [1, 360] range, create hue bands
    subset_hsv = data[['H', 'S', 'V']].astype(float)
    subset_hsv['HH'] = subset_hsv.apply(lambda row: int(float(row['H']) * 360), axis=1)
    subset_hsv['Band'] = subset_hsv.apply(lambda row: int(row['HH']) - (int(row['HH']) % divisor), axis=1)

    # count clusters per band
    counts = Counter(subset_hsv['Band'])
    counts_keys = list(counts.keys())
    for key in [k for k in ranges if k not in counts_keys]: counts[key] = 0  # pad zeroes
    for key in [k for k in ranges if 125 < k < 360]: counts[key] = 0  # remove outliers (non red/green)
    total = sum(counts.values())
    mass = OrderedDict(sorted({k: float(v / total) for k, v in counts.items()}.items()))
    mass_df = pd.DataFrame(zip([str(k) for k in mass.keys()], mass.values()), columns=['band', 'mass'])

    # radial bar plot for color distribution
    fig = px.bar_polar(
        mass_df,
        title=f"Hue distribution ({treatment})",
        r='mass',
        range_r=[0, max(mass_df['mass'])],
        theta='band',
        range_theta=[0, 360],
        color='band',
        color_discrete_map={str(k): hue_to_rgb_formatted(k) for k in counts.keys()},
        labels=None)
    fig.update_layout(showlegend=False, polar_angularaxis_tickfont_size=7, polar_radialaxis_tickfont_size=7)
    # fig.show()
    fig.write_image(join(output_directory, treatment + '.hue.radial.png'))


def color_analysis(image: np.ndarray, i: int, k: int = 15) -> Tuple[dict, np.ndarray]:
    print(f"K-means color clustering for plant {i}, k = {k}...")
    z = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((image.shape[:-1]))
    averaged = np.uint8(centers)[labels]

    print(f"Removing unwanted hues from plant {i}")
    lower_blue = np.array([0, 38, 40])
    upper_blue = np.array([39, 255, 255])
    hsv = cv2.cvtColor(averaged, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    filtered = cv2.bitwise_and(image, image, mask=mask)

    # print(f"Reapplying threshold to plant {i}")
    # mask = cv2.cvtColor(np.zeros_like(averaged), cv2.COLOR_RGB2GRAY)
    # thresholded = np.zeros_like(averaged)
    # _, thresh = cv2.threshold(cv2.cvtColor(filtered.copy(), cv2.COLOR_BGR2GRAY), 40, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(mask, contours, -1, 255, -1)
    # thresholded[mask == 255] = image[mask == 255]

    # k-means averaging
    # z = np.float32(res.reshape((-1, 3)))
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # labels = labels.reshape((res.shape[:-1]))
    # reduced = np.uint8(centers)[labels]
    # print(f"Plant {i} averaged with k = {k} clusters")

    counts = dict()
    for ii, cc in enumerate(centers):
        if all(c < 1 for c in cc):
            print(f"Plant {i} color cluster {ii} is background, ignoring")
            continue

        hex_code = rgb2hex(cc)
        mask = np.dstack([cv2.inRange(labels, ii, ii)] * 3)
        counts[hex_code] = len(np.nonzero(cv2.bitwise_and(filtered, mask))[0])
        print(f"Plant {i} color cluster {ii}: {hex_code}")

    return counts, filtered
