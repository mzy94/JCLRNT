import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from utils import next_batch_index


class MLPReg(nn.Module):
    def __init__(self, input_size, num_layers, activation):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


def data_loader(data_path, file_list, padding_id):
    min_len, max_len = 1, 100

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
    y_arr = np.zeros([num_samples], dtype=np.float32)
    for i in tqdm(range(num_samples)):
        row = df.iloc[i]
        path_arr = np.array(row['path'], dtype=np.int32)
        x_arr[i, :row['path_len']] = path_arr
        y_arr[i] = row['total_time']
    return torch.LongTensor(x_arr), torch.FloatTensor(y_arr)


def evaluation(model, data_path, file_list, num_nodes):
    print("\n--- Time Estimation ---")
    batch_size = 64
    data, y = data_loader(data_path, file_list, num_nodes)
    data_size = data.shape[0]

    model.eval()
    x = []
    for batch_idx in tqdm(next_batch_index(data_size, batch_size, shuffle=False)):
        data_batch = data[batch_idx].cuda()
        seq_rep = model.encode_sequence(data_batch)
        if isinstance(seq_rep, tuple):
            seq_rep = seq_rep[0]
        x.append(seq_rep.detach().cpu())
    x = torch.cat(x, dim=0)

    random_index = np.random.permutation(x.shape[0])
    x, y = x[random_index], y[random_index]
    split = int(data_size * 0.2)
    x, x_eval = x[split:], x[:split]
    y, y_eval = y[split:], y[:split]

    model = MLPReg(x.shape[1], 3, nn.ReLU()).cuda()
    opt = torch.optim.Adam(model.parameters())

    patience = 3
    best = [0, 1e9, 1e9]
    for e in range(1, 101):
        model.train()
        for batch_index in next_batch_index(x.shape[0], batch_size):
            opt.zero_grad()
            x_batch = x[batch_index].cuda()
            y_batch = y[batch_index].cuda()
            loss = nn.MSELoss()(model(x_batch), y_batch)
            loss.backward()
            opt.step()

        model.eval()
        y_preds = []
        for batch_index in next_batch_index(x_eval.size(0), batch_size, shuffle=False):
            x_batch = x_eval[batch_index].cuda()
            y_preds.append(model(x_batch).detach().cpu())
        y_preds = torch.cat(y_preds, dim=0)
        mae = mean_absolute_error(y_eval, y_preds)
        rmse = mean_squared_error(y_eval, y_preds) ** 0.5
        print(f'Epoch: {e}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')
        if mae < best[1]:
            best = [e, mae, rmse]
            patience = 3
        else:
            if e > 10:
                patience -= 1
            if not patience:
                print(f'Best epoch: {best[0]}, MAE: {best[1].item():.4f}, RMSE: {best[2].item():.4f}')
                break
    return best
