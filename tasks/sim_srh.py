import os
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm

from utils import next_batch_index


def data_loader(data_path, file_list, padding_id, num_queries, detour_rate=0.1):
    min_len, max_len = 20, 100

    dfs = []
    for file_name in file_list:
        tmp_df = pd.read_csv(os.path.join(data_path, file_name))
        dfs.append(tmp_df)
    df = pd.concat(dfs).reset_index(drop=True)
    df['path'] = df['path'].map(eval)
    df['path_len'] = df['path'].map(len)
    df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]

    num_samples = len(df)
    x_arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(num_samples)):
        row = df.iloc[i]
        path_arr = np.array(row['path'], dtype=np.int32)
        x_arr[i, :row['path_len']] = path_arr

    def detour(rate=0.9):
        p = np.random.random_sample()
        return np.random.randint(padding_id) if p > rate else padding_id

    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(num_queries)):
        row = df.iloc[random_index[i]]
        detour_pos = np.random.choice(row['path_len'], int(row['path_len'] * detour_rate), replace=False)
        path = [detour() if i in detour_pos else e for i, e in enumerate(row['path'])]
        q_arr[i, :row['path_len']] = np.array(path, dtype=np.int32)

    y = random_index[:num_queries]
    return torch.LongTensor(x_arr), torch.LongTensor(q_arr), y


def evaluation(model, data_path, file_list, num_nodes):
    print("\n--- Similarity Search ---")
    batch_size = 64
    num_queries = 5000
    data, queries, y = data_loader(data_path, file_list, num_nodes, num_queries)
    data_size = data.shape[0]

    model.eval()
    x = []
    for batch_idx in tqdm(next_batch_index(data_size, batch_size, shuffle=False)):
        data_batch = data[batch_idx].cuda()
        seq_rep = model.encode_sequence(data_batch)
        if isinstance(seq_rep, tuple):
            seq_rep = seq_rep[0]
        x.append(seq_rep.detach().cpu())
    x = torch.cat(x, dim=0).numpy()

    q = []
    for batch_idx in tqdm(next_batch_index(num_queries, batch_size, shuffle=False)):
        q_batch = queries[batch_idx].cuda()
        seq_rep = model.encode_sequence(q_batch)
        if isinstance(seq_rep, tuple):
            seq_rep = seq_rep[0]
        q.append(seq_rep.detach().cpu())
    q = torch.cat(q, dim=0).numpy()

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    D, I = index.search(q, 10000)
    hit = 0
    rank_sum = 0
    no_hit = 0
    for i, r in enumerate(I):
        if y[i] in r:
            rank_sum += np.where(r == y[i])[0][0]
            if y[i] in r[:10]:
                hit += 1
        else:
            no_hit += 1
    print(f'Mean Rank: {rank_sum / num_queries}, HR@10: {hit / (num_queries - no_hit)}, No Hit: {no_hit}')
