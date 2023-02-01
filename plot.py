import math

import networkx as nx
import numpy as np
from networkx import *
import matplotlib.pyplot as plt

from causalFM.plot_config import arrow_active_color, arrow_alternative_color, arrow_inactive_color, \
    arrow_positive_color, arrow_negative_color, arrow_unknown_color

var_positions = {
    # ALTITUDE
    "altitude": [0, 0],
    "temperature": [1, 0],
    
    # CAUSAL HEALTH
    "age": [0, 0],
    "health": [0.5, 0.5],
    "mobility": [0.5, 1],
    "nutrition": [1, 0],
    
    # DRIVING
    "fuel": [0.5, 1],
    "driveStyle": [1, 0],
    "carType": [0, 0],
    
    # RECOVERY
    "treatment": [0, 0.5],
    "recovery": [1, 0.5],
    "precondition": [0.5, 0],

    # EARTHQUAKE
    "earthquake": [0, 0],
    "burglaries": [1, 0],
    "alarms": [0.5, 0.3],
    "john": [0, 1],
    "mary": [1, 1],

    # CANCER
    "pollution": [0, 0],
    "smoking": [1, 0],
    "cancer": [0.5, 0.3],
    "x-ray": [0, 1],
    "dyspnoea": [1, 1]
}

plot_sizes = {
    "altitude": [1, 0.1],
    "causal_health": [1, 0.1],
    "driving": [1, 0.1],
    "recovery": [1, 0.1],
    "cancer": [1, 0.1],
    "earthquake": [1, 0.1]
}


def lerp(a, b, x):
    return (a*(1-x)) + (b*x)


def draw_edge(p0, p1, f01, f10, pad, active_color=None, alt_color=None, mode=None,
              label=None, label0=None, label1=None, label_x_percent=0.5,
              color_p=None, color_n=None, color_u=None):
    """
    :param p0: node0 position
    :param p1: node1 position
    :param f01: adj_mat[i0,i1]
    :param f10: adj_mat[i1,i0]
    :param pad: node radius
    :param active_color: active arrow color
    :param alt_color: alternative color for diverging mode
    :param mode: "encode" (1=has edge, 0=no edge, -1=unknown), "strength" (0.0=invisible, 1.0=dark),
    "diverging" (positive_active_color, negative=alt_color, zero=merge, nan=hidden)
    :param label: places text at the center of the label
    :param label0: places text near p0
    :param label1: places text near p1
    :param label_x_percent: determines where label0/1 are placed (0.0=center, 1.0=at tip of arrow)
    :return:
    """
    if active_color is None:
        active_color = arrow_active_color
    if alt_color is None:
        alt_color = arrow_alternative_color
    if color_p is None:
        color_p = arrow_positive_color
    if color_n is None:
        color_n = arrow_negative_color
    if color_u is None:
        color_u = arrow_unknown_color
    inactive_color = arrow_inactive_color

    if mode is None:
        mode = "encode"

    dir = [p1[0] - p0[0], p1[1] - p0[1]]
    l = math.sqrt(dir[0]**2 + dir[1]**2)
    lp = l - 2*pad
    ndir = [dir[0]/l, dir[1]/l]

    start = [p0[0] + ndir[0]*pad, p0[1] + ndir[1] * pad]
    end = [p1[0] - ndir[0]*pad, p1[1] - ndir[1] * pad]
    center = [
        (start[0]+end[0])/2,
        (start[1]+end[1])/2
        ]

    if mode == "encode":
        #FIXME handle -1 edges (draw them grey?)
        #1=edge, 0=no edge, -1=unknown
        a01 = f01 == 1
        a10 = f10 == 1
        has_connection = a01 or a10
        if not has_connection:
            return
        r_color01 = active_color
        r_color10 = active_color
    elif mode == "sign":
        a01 = True
        a10 = True
        r_color01 = color_p if f01 > 0 else color_u if f01 < 0 else color_n
        r_color10 = color_p if f10 > 0 else color_u if f10 < 0 else color_n
    elif mode == "strength":
        a01 = f01 != 0
        a10 = f10 != 0
        has_connection = a01 or a10
        if not has_connection:
            return
        c0, c1, c2 = active_color
        b0, b1, b2 = inactive_color

        f01s = 1 - f01
        f10s = 1 - f10
        r_color01 = (lerp(b0, c0, f01s), lerp(b1, c1, f01s), lerp(b2, c2, f01s))
        r_color10 = (lerp(b0, c0, f10s), lerp(b1, c1, f10s), lerp(b2, c2, f10s))
    elif mode == "diverging":
        has_no_connection = np.isnan(f01) or np.isnan(f10)
        # handle case where only one edge exists
        assert np.isnan(f01) == np.isnan(f10)

        if has_no_connection:
            return
        a01 = True
        a10 = True

        f01 = (f01 + 1) / 2  # map -1..1 to 0..1
        f10 = (f10 + 1) / 2

        f01s = 1 - f01
        f10s = 1 - f10
        c0, c1, c2 = active_color #->positive
        b0, b1, b2 = alt_color #->negative
        r_color01 = (lerp(b0, c0, f01s), lerp(b1, c1, f01s), lerp(b2, c2, f01s))
        r_color10 = (lerp(b0, c0, f10s), lerp(b1, c1, f10s), lerp(b2, c2, f10s))
    else:
        raise RuntimeError(f"Unknown display mode: {mode}")

    plt.arrow(
        #start[0], start[1], ndir[0]*(l-2*pad), ndir[1]*(l-2*pad),
        center[0], center[1], ndir[0]*(0.5*l-1*pad), ndir[1]*(0.5*l-1*pad),
        width=0.005,
        head_width=0.05 if a01 else 0,
        length_includes_head=True,
        edgecolor=r_color01,
        facecolor=r_color01
    )

    plt.arrow(
        #end[0], end[1], -ndir[0]*(l-2*pad), -ndir[1]*(l-2*pad),
        center[0], center[1], -ndir[0]*(0.5*l-1*pad), -ndir[1]*(0.5*l-1*pad),
        width=0.005,
        head_width=0.05 if a10 else 0,
        length_includes_head=True,
        edgecolor=r_color10,
        facecolor=r_color10
    )

    if label is not None:
        plt.text(center[0], center[1], label, horizontalalignment='center', verticalalignment="center")
    if label0 is not None:
        plt.text(
            center[0] + ndir[0] * label_x_percent * -lp/2,
            center[1] + ndir[1] * label_x_percent * -lp/2,
            label0, horizontalalignment='center', verticalalignment="center")
    if label1 is not None:
        plt.text(
            center[0] + ndir[0] * label_x_percent * lp/2,
            center[1] + ndir[1] * label_x_percent * lp/2,
            label1, horizontalalignment='center', verticalalignment="center")

def plot_from_adj_mat(
        adj_mat, var_names, dataset_name, ax=None, abrev_vars=True, ignore_undefined=None,
        edge_labels=None, edge_mode=None):
    """

    :param adj_mat: 2d-array [i,j]=1 = i->j
    :param var_names:
    :param dataset_name:
    :param ax:
    :param abrev_vars:
    :param ignore_undefined:
    :param edge_labels:
    :param edge_mode:
    :return:
    """
    if ax is None:
        raise RuntimeError("ax is none")

    positions = []
    pos_by_name = {}
    for i, var_name in enumerate(var_names):
        pos = var_positions[var_name].copy()
        pos[1] = -pos[1]
        positions.append(pos)
        pos_by_name[var_name] = pos

    var_labels = [var_name[0] if abrev_vars else var_name for var_name in var_names]
    var_labels = [var_label.capitalize() for var_label in var_labels]

    if ignore_undefined is not None:
        print("[deprecated] 'ignore_undefined' argument has no effect")

    circle_radius = 0.1

    # draw edges
    num_vars = len(var_names)
    for i0, v0 in enumerate(var_names):
        for i1, v1 in enumerate(var_names):
            if i1 <= i0:
                continue
            e01 = adj_mat[i0, i1]
            e10 = adj_mat[i1, i0]

            p0 = pos_by_name[v0]
            p1 = pos_by_name[v1]

            label0 = None if edge_labels is None else edge_labels[i0][i1]
            label1 = None if edge_labels is None else edge_labels[i1][i0]
            draw_edge(p0, p1, e01, e10, circle_radius, label0=label1, label1=label0, mode=edge_mode)


    minx = 1000
    miny = 1000
    maxx = -1000
    maxy = -1000

    # draw nodes
    for i, var_name in enumerate(var_names):
        pos = pos_by_name[var_name]

        minx = min(minx, pos[0])
        miny = min(miny, pos[1])
        maxx = max(maxx, pos[0])
        maxy = max(maxy, pos[1])

        circle = plt.Circle(pos, circle_radius, edgecolor='black', fill=True, facecolor='lightgrey')
        ax.add_patch(circle)

        plt.text(pos[0], pos[1], var_labels[i], horizontalalignment='center', verticalalignment="center")
        #ax.add_patch(text)

    pad = 0.15
    if dataset_name is None:
        raise RuntimeError("not supported")
    else:
        plot_size = plot_sizes[dataset_name]
        ax.set_xlim(minx - pad, minx + plot_size[0] + pad)
        if maxy - miny < 0.01:
            ax.set_ylim(- pad, + pad)
        else:
            ax.set_ylim(miny - pad, minx + plot_size[1] + pad)
