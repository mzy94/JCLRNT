import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm


def train_data_loader(data_path, file_list, padding_id, config):
    min_len = config['min_len']
    max_len = config['max_len']
    num_samples = config['num_samples']

    dfs = []
    for file_name in file_list:
        tmp_df = pd.read_csv(os.path.join(data_path, file_name))
        dfs.append(tmp_df)
    df = pd.concat(dfs).reset_index(drop=True)
    df['path'] = df['path'].map(eval)
    df['path_len'] = df['path'].map(len)
    df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]
    if len(df) > num_samples:
        df = df.sample(n=num_samples, replace=False, random_state=1)

    arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        path_arr = np.array(row['path'], dtype=np.int32)
        arr[i, :row['path_len']] = path_arr

    if config['weighted_loss']:
        weights = []
        for file_name in tqdm(file_list):
            weight = np.load(os.path.join(data_path, file_name[4:-4] + '_w.npy'))
            weights.append(weight)
        weights = np.concatenate(weights, axis=0)[df.index.values]
        assert arr.shape[0] == weights.shape[0]
        weights[weights < config['weight_threshold']] = 0
        weights[weights > 1] = 1
        return torch.LongTensor(arr), torch.tensor(weights)
    else:
        return torch.LongTensor(arr), None


def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)

    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index


def top_k_acc(output, target, ks=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(ks)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    if m == 0 and n != 0:
        return n, 1 - n / max(m, n)
    elif m != 0 and n == 0:
        return m, 1 - m / max(m, n)
    elif m == 0 and n == 0:
        try:
            1 - 0 / 0
        except ZeroDivisionError as z:
            print(z)
        return math.nan, math.nan
    else:
        d = np.zeros((n + 1, m + 1))
        d[0] = np.arange(m + 1)
        d[:, 0] = np.arange(n + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[j - 1] == seq2[i - 1]:
                    temp = 0
                else:
                    temp = 1
                d[i, j] = min(d[i - 1, j] + 1, 
                              d[i, j - 1] + 1, 
                              d[i - 1, j - 1] + temp)
        return d[n, m], 1 - d[n, m] / max(m, n)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        # init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5 / m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)
    elif isinstance(m, nn.Bilinear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
