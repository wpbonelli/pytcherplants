from typing import Dict, Tuple

import pandas as pd
import seaborn as sns
from plotly import graph_objects as go, express as px

from pytcherplants.utils import rgb2hex, hue_to_rgb_formatted


def plot_hex_distribution(proportions: Dict[Tuple[int, int, int], float], title: str):
    x = list([rgb2hex(k).replace('-', '') for k in proportions.keys()])
    y = list(proportions.values())
    ax = sns.histplot(x=x, weights=y, hue=x, palette=x, discrete=True)
    ax.set_title(title)
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


def plot_hue_distribution(proportions: Dict[Tuple[int, int, int], float], title: str) -> go.Figure:
    df = pd.DataFrame(zip([str(k) for k in proportions.keys()], proportions.values()), columns=['bin', 'mass'])
    fig = px.bar_polar(
        df,
        title=f"Hue distribution ({title})",
        r='mass',
        range_r=[0, max(df['mass'])],
        theta='bin',
        range_theta=[0, 360],
        color='bin',
        color_discrete_map={str(k): hue_to_rgb_formatted(k) for k in proportions.keys()},
        labels=None)
    fig.update_layout(showlegend=False, polar_angularaxis_tickfont_size=7, polar_radialaxis_tickfont_size=7)
    return fig
