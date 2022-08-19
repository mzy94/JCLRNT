import os
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from multiprocessing import Pool
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from fmm import Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT, UBODTGenAlgorithm
from tqdm import tqdm


def gcj2wgs(point):
    lon, lat = point
    a = 6378245.0  # 克拉索夫斯基椭球参数长半轴a
    ee = 0.00669342162296594323  # 克拉索夫斯基椭球参数第一偏心率平方
    pi = 3.14159265358979324  # 圆周率
    # 以下为转换公式
    x = lon - 105.0
    y = lat - 35.0
    # 经度
    delta_lon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    delta_lon += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    delta_lon += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
    delta_lon += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
    # 纬度
    delta_lat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    delta_lat += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    delta_lat += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
    delta_lat += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
    rad_lat = lat / 180.0 * pi
    magic = math.sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = math.sqrt(magic)
    delta_lat = (delta_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * pi)
    delta_lon = (delta_lon * 180.0) / (a / sqrt_magic * math.cos(rad_lat) * pi)
    wgs_lon = lon - delta_lon
    wgs_lat = lat - delta_lat
    return wgs_lon, wgs_lat


def get_map_range(file_name):
    df = pd.read_csv(os.path.join(raw_data_path, file_name),
                     names=['driver_id', 'order_id', 'timestamp', 'lon', 'lat'])
    wgs_coordinates = [gcj2wgs(p) for p in zip(df['lon'], df['lat'])]
    wgs_lons = [p[0] for p in wgs_coordinates]
    wgs_lats = [p[1] for p in wgs_coordinates]
    return [max(wgs_lats), min(wgs_lats), max(wgs_lons), min(wgs_lons)]


def graph_from_bbox(bbox,
                    place_name=None,
                    network_type='drive',
                    truncate_by_edge=True,
                    cached=True,
                    cache_path='./osm_graph'):
    if not place_name:
        place_name = str(bbox)
    pkl_file = os.path.join(cache_path, place_name + '.pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as bf:
            graph = pickle.load(bf)
    else:
        n, s, e, w = bbox
        graph = ox.graph_from_bbox(n, s, e, w,
                                   network_type=network_type,
                                   truncate_by_edge=truncate_by_edge)
        if cached:
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            with open(pkl_file, 'wb') as bf:
                pickle.dump(graph, bf)
    return graph


def save_graph_shapefile(graph, file_path, encoding="utf-8"):
    # if save folder does not already exist, create it (shapefiles get saved as set of files)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    filepath_nodes = os.path.join(file_path, "nodes.shp")
    filepath_edges = os.path.join(file_path, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(graph)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = gdf_edges.index
    gdf_edges["fid"] = gdf_edges["fid"].map(edge2idx)
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


def agg_traj(df):
    df = df.sort_values('timestamp')
    time = df['timestamp'].tolist()
    line = list(zip(df['lon'], df['lat']))
    return pd.Series({'timestamp': time, 'polyline': line})


def process_row(inputs):
    order_id, row = inputs
    timestamp = row['timestamp']
    polyline = [gcj2wgs(p) for p in row['polyline']]
    if len(polyline) > 1:
        st = timestamp[0]
        et = timestamp[-1]
        tt = et - st
        wkt = str(LineString(polyline))
        result = model.match_wkt(wkt, config)
        path = [e for e in list(result.cpath)]
        if len(path) > 2 and tt > 60:
            edges = [idx2edge[i] for i in path]
            nodes = [e[0] for e in edges] + [edges[-1][1]]
            node_pos = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes]
            dist_mat = cdist(np.array(node_pos), np.array(polyline))
            min_idxs = dist_mat.argmin(axis=1)
            times = [timestamp[idx] for idx in min_idxs]
            pt = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
            return order_id, st, str(path), str(times), str(pt), tt
        else:
            return None
    else:
        return None


def process_file(file_name):
    raw_df = pd.read_csv(file_name, names=['driver_id', 'order_id', 'timestamp', 'lon', 'lat'])
    order_df = raw_df.groupby('order_id').apply(agg_traj)

    records = []
    pool = Pool(4)
    for result in tqdm(pool.imap(process_row, order_df.iterrows())):
        if result:
            records.append(result)
    pool.close()
    pool.join()

    return pd.DataFrame(records, columns=['order_id', 'start_time', 'path', 'timestamp', 'pass_time', 'total_time'])


def get_road_speed(files):
    speed_dict = {}
    rs_length = feature_df['length']
    for file_name in files:
        data_df = pd.read_csv(os.path.join(data_path, file_name))
        for _, row in tqdm(data_df.iterrows()):
            path = eval(row['path'])
            pt = eval(row['pass_time'])
            for i, rs in enumerate(path):
                if pt[i] > 0:
                    tmp = rs_length[rs] / pt[i]
                    if tmp < 30:
                        avg, n = speed_dict.get(rs, (0, 0))
                        speed_dict[rs] = ((avg * n + tmp) / (n + 1), (n + 1))
    return [speed_dict.get(i, (0, 0))[0] for i in range(len(feature_df))]


def get_traj_speed(files):
    speed_dict = {}
    rs_length = feature_df['length']
    for file_name in files:
        data_df = pd.read_csv(os.path.join(data_path, file_name))
        for _, row in tqdm(data_df.iterrows()):
            path = eval(row['path'])
            tl = sum([rs_length[rs] for rs in path])
            tmp = tl / row['total_time']
            for i, rs in enumerate(path):
                avg, n = speed_dict.get(rs, (0, 0))
                speed_dict[rs] = ((avg * n + tmp) / (n + 1), (n + 1))
    return [speed_dict.get(i, (0, 0))[0] for i in range(len(feature_df))]


def get_trans_mat(files):
    mat = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for file_name in files:
        pdf = pd.read_csv(os.path.join(data_path, file_name))
        for path in tqdm(pdf['path'].map(eval)):
            for i, m in enumerate(path):
                for n in path[i:]:
                    mat[m][n] += 1
    mat = mat / (mat.max(axis=1, keepdims=True, initial=0.) + 1e-9)
    row, col = np.diag_indices_from(mat)
    mat[row, col] = 0
    return mat


if __name__ == '__main__':
    # Didi dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='XiAn')
    args = parser.parse_args()
    city = args.city
    map_bbox = {'XiAn': [34.282, 34.206, 108.995, 108.906],
                'ChengDu': [30.730, 30.6554, 104.127, 104.0397]}[city]
    dates = {'XiAn': '2016110', 'ChengDu': '2016101'}[city]
    dir_name = {'XiAn': 'didi_xian', 'ChengDu': 'didi_chengdu'}[city]
    raw_data_path = os.path.join('raw_data', dir_name)
    data_path = os.path.join('datasets', dir_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Read graph data
    print('Read road network graph.')
    G = graph_from_bbox(map_bbox, city)
    print("Graph: nodes {} edges {}".format(len(list(G.nodes)), len(list(G.edges))))
    dict_path = os.path.join(data_path, 'dicts.pkl')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            dicts = pickle.load(f)
        edge2idx = dicts['edge2idx']
        idx2edge = dicts['idx2edge']
    else:
        edge_list = list(G.edges)
        edge2idx = {e: i for i, e in enumerate(edge_list)}
        idx2edge = {i: e for i, e in enumerate(edge_list)}
        with open(dict_path, 'wb') as f:
            pickle.dump({'edge2idx': edge2idx, 'idx2edge': idx2edge}, f)

    # Generate line graph
    print('Transfer network into line graph.')
    L = nx.line_graph(G, nx.DiGraph)
    lg_edges = [(edge2idx[e[0]], edge2idx[e[1]]) for e in L.edges]
    lg_edge_idx = np.array(lg_edges).transpose()
    lg_path = os.path.join(data_path, 'line_graph_edge_idx.npy')
    np.save(lg_path, lg_edge_idx)

    # Save edge features
    print('Get road segment features.')
    feature_path = os.path.join(data_path, 'edge_features.csv')
    features = []
    for k, v in idx2edge.items():
        oneway = G.edges[v].get('oneway', False)
        lanes = G.edges[v].get('lanes', '0')
        highway = G.edges[v].get('highway', 'unclassified')
        length = G.edges[v].get('length', 0)
        bridge = 1 if 'bridge' in G.edges[v] else 0
        tunnel = 1 if 'tunnel' in G.edges[v] else 0
        features.append((k, oneway, lanes, highway, length, bridge, tunnel))
    feature_df = pd.DataFrame(features, columns=[
        'road_id', 'oneway', 'lanes', 'highway', 'length', 'bridge', 'tunnel'])
    feature_df['oneway'] = feature_df['oneway'].map(lambda x: int(x) if type(x) != list else 0)
    feature_df['lanes'] = feature_df['lanes'].map(lambda x: eval(x[0]) if type(x) == list else eval(x))
    feature_df['highway'] = feature_df['highway'].map(lambda x: x[0] if type(x) == list else x)
    tmp_feat = feature_df.loc[feature_df['highway'] != 'unclassified']['highway']
    highway2idx = {hw: i + 1 for i, hw in enumerate(tmp_feat.value_counts().index)}
    highway2idx['unclassified'] = 0
    feature_df['highway_id'] = feature_df['highway'].map(highway2idx)
    feature_df['length_id'] = feature_df['length'].map(lambda x: math.ceil(x / 100))
    feature_df.sort_values('road_id').reset_index(inplace=True)
    # feature_df.to_csv(feature_path, index=False)

    print('prepare map matching algorithm.')
    # Read network data
    shapefile_path = os.path.join('shapefile', city)
    if not os.path.exists(os.path.join(shapefile_path, 'edges.shp')):
        save_graph_shapefile(G, shapefile_path, encoding="utf-8")
    shapefile = os.path.join(shapefile_path, "edges.shp")
    network = Network(shapefile, "fid", "u", "v")
    network_graph = NetworkGraph(network)
    print("Network: Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))

    # Read UBODT data
    ubodt_file = os.path.join(shapefile_path, "ubodt.txt")
    if not os.path.exists(ubodt_file):
        ubodt_gen = UBODTGenAlgorithm(network, network_graph)
        status = ubodt_gen.generate_ubodt(ubodt_file, 0.02, binary=False, use_omp=True)
    ubodt = UBODT.read_ubodt_csv(ubodt_file)

    # Define map matching model and configurations
    model = FastMapMatch(network, network_graph, ubodt)
    k = 8
    radius = 0.003
    gps_error = 0.0005
    config = FastMapMatchConfig(k, radius, gps_error)

    # Processing and save
    print('Run map matching.')
    raw_files = [f for f in os.listdir(raw_data_path) if f.startswith('gps_' + dates)]
    for f in raw_files:
        if not os.path.exists(os.path.join(data_path, f[4:] + '.csv')):
            print(f"Processing file: {f}")
            processed_df = process_file(os.path.join(raw_data_path, f))
            processed_df.to_csv(os.path.join(data_path, f[4:] + '.csv'), index=False)

    num_nodes = len(G.edges)
    new_files = [f for f in os.listdir(data_path) if f.startswith(dates)]

    # traffic speed
    feature_df['road_speed'] = get_road_speed(new_files)
    feature_df['traj_speed'] = get_traj_speed(new_files)
    feature_df.to_csv(feature_path, index=False)

    # Calculate transition probability matrix
    print('Get transition probability matrix.')
    trans_mat = get_trans_mat(new_files)
    np.save(os.path.join(data_path, 'transition_prob_mat.npy'), trans_mat)

    print('Done.')
