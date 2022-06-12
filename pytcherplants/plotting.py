from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly import graph_objects as go, express as px

from pytcherplants.utils import rgb2hex


def plot_hex_distribution(proportions: Dict[Tuple[int, int, int], float], title: str):
    x = list([rgb2hex(k).replace('-', '') for k in proportions.keys()])
    y = list(proportions.values())
    ax = sns.histplot(x=x, weights=y, hue=x, palette=x, discrete=True)
    ax.set_title(title)
    plt.ylabel('Proportion')
    return ax


def plot_rgb_distribution(proportions: Dict[Tuple[int, int, int], float], title: str) -> go.Figure:
    fig = go.Figure()
    r = [k[0] for k in proportions.keys()]
    g = [k[1] for k in proportions.keys()]
    b = [k[2] for k in proportions.keys()]
    colors_map = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in proportions.keys()]
    sizes_map = list([v * 1000 for v in proportions.values()])
    trace = dict(type='scatter3d', x=r, y=g, z=b, mode='markers', marker=dict(color=colors_map, size=sizes_map))
    fig.add_trace(trace)
    fig.update_layout(title=title, scene=dict(xaxis_title='G', yaxis_title='R', zaxis_title='B'))
    return fig


def plot_hue_distribution(proportions: Dict[int, float], title: str) -> go.Figure:
    df = pd.DataFrame(zip([str(k) for k in proportions.keys()], proportions.values()), columns=['bin', 'mass'])
    cmap = {str(k): f"hsv({str(int(k / 360 * 100))}%, 50%, 50%)" for k in proportions.keys()}
    fig = px.bar_polar(
        df,
        title=title,
        r='mass',
        range_r=[0, max(df['mass'])],
        theta='bin',
        range_theta=[0, 360],
        color='bin',
        color_discrete_map=cmap,
        labels=None)
    fig.update_layout(showlegend=False, polar_angularaxis_tickfont_size=7, polar_radialaxis_tickfont_size=7)
    return fig


# referenced from https://stackoverflow.com/a/55067613
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches
