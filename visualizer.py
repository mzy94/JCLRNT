import os
import pickle
import numpy as np
import osmnx as ox
from osmnx import utils_geo
from geopy.distance import distance


def graph_from_place(place_name,
                     network_type='drive',
                     cache_path='./osm_graph',
                     cached=True):
    pkl_file = os.path.join(cache_path, place_name + '.pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            graph = pickle.load(f)
    else:
        graph = ox.graph_from_place(place_name, network_type=network_type)
        if cached:
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            with open(pkl_file, 'wb') as f:
                pickle.dump(graph, f)
    return graph


def graph_from_points(points, max_radius=None, network_type='drive', return_graph=False):
    pos_array = np.array(points)
    center_lon = pos_array[:, 0].mean()
    center_lat = pos_array[:, 1].mean()
    dist_from_center = np.apply_along_axis(
        lambda x: distance((x[1], x[0]), (center_lat, center_lon)).m, axis=1, arr=pos_array)
    radius = dist_from_center.max() * 1.05
    # print('graph radius:', radius)
    if max_radius:
        radius = min(radius, max_radius)
    bbox = utils_geo.bbox_from_point((center_lat, center_lon), radius)
    if return_graph:
        north, south, east, west = bbox
        graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
        return bbox, graph
    return bbox


def plot_graph(graph,
               figsize=(20, 20),
               dpi=100,
               bgcolor="w",
               node_color="y",
               node_size=2,
               show=False,
               close=False):
    fig, ax = ox.plot_graph(graph,
                            figsize=figsize,
                            dpi=dpi,
                            bgcolor=bgcolor,
                            node_color=node_color,
                            node_size=node_size,
                            show=show,
                            close=close)
    return fig, ax


def plot_line(line, ax, color='b', line_width=1, line_style='-'):
    lngs = [p[0] for p in line]
    lats = [p[1] for p in line]
    ax.plot(lngs, lats, linestyle=line_style, linewidth=line_width, color=color)


def plot_points(points, ax, c='b', s=30):
    lngs = [p[0] for p in points]
    lats = [p[1] for p in points]
    ax.scatter(lngs, lats, c=c, s=s)
