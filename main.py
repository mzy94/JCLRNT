import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from models import sv, mv
from tasks import road_cls, speed_inf, time_est, sim_srh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='XiAn')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--num_train', type=int, default=500000)
    parser.add_argument('--edge_threshold', type=float, default=0.6)
    parser.add_argument('--lambda_st', type=float, default=1.)
    parser.add_argument('--model', type=str, default='sv')
    parser.add_argument('--mode', type=str, default='s')
    parser.add_argument('--is_weighted', type=bool, default=False)
    parser.add_argument('--retrain', type=bool, default=False)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    city = args.city
    dates = {'XiAn': '2016110', 'ChengDu': '2016101', 'Porto': '2013'}[city]
    dir_name = {'XiAn': 'didi_xian', 'ChengDu': 'didi_chengdu', 'Porto': 'porto_taxi'}[city]
    data_path = os.path.join('datasets', dir_name)
    save_path = os.path.join('checkpoints', dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    feature_file = os.path.join(data_path, 'edge_features.csv')
    feature_df = pd.read_csv(feature_file)
    num_nodes = len(feature_df)
    edge_index_file = os.path.join(data_path, 'line_graph_edge_idx.npy')
    edge_index = torch.tensor(np.load(edge_index_file)).cuda()
    
    trans_mat_file = os.path.join(data_path, 'transition_prob_mat.npy')
    trans_mat = np.load(trans_mat_file)
    trans_mat_b = (trans_mat > args.edge_threshold)
    aug_edges = [(i // num_nodes, i % num_nodes) for i, n in enumerate(trans_mat_b.flatten()) if n]
    aug_edge_index = torch.tensor(np.array(aug_edges).transpose()).cuda()

    data_files = sorted([f for f in os.listdir(data_path) if f.startswith(dates) and f.endswith('.csv')])
    train_data_files = data_files[:-1]
    task_data_files = data_files[-1:]

    config['num_samples'] = args.num_train
    config['lambda_st'] = args.lambda_st
    config['weighted_loss'] = args.is_weighted
    config['retrain'] = args.retrain
    config['mode'] = args.mode

    if args.model == 'sv':
        model = sv.train(data_path, save_path, train_data_files, num_nodes, edge_index, config)

    if args.model == 'mv':
        model = mv.train(data_path, save_path, train_data_files, num_nodes, edge_index, aug_edge_index, config)

    print("\n=== Evaluation ===")
    # road-based tasks
    road_cls.evaluation(model, feature_df)
    speed_inf.evaluation(model, feature_df)
    # trajectory-based tasks
    time_est.evaluation(model, data_path, task_data_files, num_nodes)
    sim_srh.evaluation(model, data_path, task_data_files, num_nodes)
