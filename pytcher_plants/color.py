from collections import Counter, OrderedDict
from os.path import join
from pprint import pprint

import pandas as pd
import seaborn as sns
from scipy.cluster.vq import kmeans2
from plotly import express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pytcher_plants.utils import hue_to_rgb_formatted, rgb2hex


def rgb_analysis(treatment, output_directory, subset):
    subset_rgb = subset[['R', 'G', 'B']].astype(float).values.tolist()
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
    plt.clf()

    fig = go.Figure()
    r = [k[0] for k in props.keys()]
    g = [k[1] for k in props.keys()]
    b = [k[2] for k in props.keys()]
    colors_map = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in props.keys()]
    sizes_map = list([v * 1000 for v in props.values()])
    trace=dict(type='scatter3d', x=r, y=g, z=b, mode='markers', marker=dict(color=colors_map, size=sizes_map))
    fig.add_trace(trace)
    fig.update_layout(title=treatment, scene=dict(xaxis_title='G', yaxis_title='R', zaxis_title='B'))
    fig.write_image(join(output_directory, treatment + '.rgb.3d.png'))


def hsv_analysis(treatment, output_directory, subset):
    ranges = {((k * 5) + 5):list(range(k * 5, (k * 5) + 5)) for k in range(0, 72)}
    ranges_round = {min(v):k for k, v in ranges.items()}

    # format HSV columns, convert to [1, 360] range, create hue bands (72 equally spaced from 5 to 355)
    subset_hsv = subset[['H', 'S', 'V']].astype(float)
    subset_hsv['HH'] = subset_hsv.apply(lambda row: int(float(row['H']) * 360), axis=1)
    subset_hsv['Band'] = subset_hsv.apply(lambda row: ranges_round[round(int(row['HH']), -1)], axis=1)

    # count clusters per band
    counts = Counter(subset_hsv['Band'])
    counts_keys = list(counts.keys())
    for key in [k for k in ranges.keys() if k not in counts_keys]: counts[key] = 0  # pad zeroes
    for key in [k for k in ranges.keys() if 125 < k < 360]: counts[key] = 0         # remove outliers (non red/green)
    total = sum(counts.values())
    mass = OrderedDict(sorted({k:float(v / total) for k, v in counts.items()}.items()))
    mass_df = pd.DataFrame(zip([str(k) for k in mass.keys()], mass.values()), columns=['band', 'mass'])

    # radial bar plot for color distribution
    fig = px.bar_polar(
        mass_df,
        title=f"Hue distribution ({treatment})",
        r='mass',
        range_r=[0, 0.5], # max(mass_df['mass'])
        theta='band',
        range_theta=[0,360],
        color='band',
        color_discrete_map={str(k): hue_to_rgb_formatted(k) for k in counts.keys()},
        labels=None)
    fig.update_layout(showlegend=False, polar_angularaxis_tickfont_size=7, polar_radialaxis_tickfont_size=7)
    fig.write_image(join(output_directory, treatment + '.hue.radial.png'))