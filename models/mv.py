import os
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import utils


def jsd(z1, z2, pos_mask):
    neg_mask = 1 - pos_mask

    sim_mat = torch.mm(z1, z2.t())
    E_pos = math.log(2.) - F.softplus(-sim_mat)
    E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.)
    return (E_neg * neg_mask).sum() / neg_mask.sum() - (E_pos * pos_mask).sum() / pos_mask.sum()


def nce(z1, z2, pos_mask):
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()


def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    return -torch.log((sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)).mean()


def node_node_loss(node_rep1, node_rep2, measure):
    num_nodes = node_rep1.shape[0]

    pos_mask = torch.eye(num_nodes).cuda()

    if measure == 'jsd':
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == 'nce':
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).cuda()

    if measure == 'jsd':
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).cuda()
    for row_idx, row in enumerate(sequences):
        pos_mask[row_idx][row] = 1.
    pos_mask = pos_mask[:, :-1]

    if measure == 'jsd':
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    if measure == 'jsd':
        return jsd(seq_rep, node_rep, weights)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, weights)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, weights)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = [encoder_layer(input_size, output_size)]
        for _ in range(1, num_layers):
            self.layers.append(encoder_layer(output_size, output_size))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.activation(self.layers[i](x, edge_index))
        return x


class MultiViewModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, edge_index1, edge_index2,
                 graph_encoder1, graph_encoder2, seq_encoder):
        super(MultiViewModel, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False).cuda()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.graph_encoder1 = graph_encoder1
        self.graph_encoder2 = graph_encoder2
        self.seq_encoder = seq_encoder

    def encode_graph(self):
        node_emb = self.node_embedding.weight
        node_enc1 = self.graph_encoder1(node_emb, self.edge_index1)
        node_enc2 = self.graph_encoder2(node_emb, self.edge_index2)
        return node_enc1 + node_enc2, node_enc1, node_enc2

    def encode_sequence(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()

        batch_size, max_seq_len = sequences.size()
        src_key_padding_mask = (sequences == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding], 0)
        seq_emb1 = torch.index_select(
            lookup_table1, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        seq_enc1 = self.seq_encoder(seq_emb1, None, src_key_padding_mask)
        seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding], 0)
        seq_emb2 = torch.index_select(
            lookup_table2, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        seq_enc2 = self.seq_encoder(seq_emb2, None, src_key_padding_mask)
        seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_pooled1 + seq_pooled2, seq_pooled1, seq_pooled2

    def forward(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()
        _, seq_pooled1, seq_pooled2 = self.encode_sequence(sequences)
        return node_enc1, node_enc2, seq_pooled1, seq_pooled2


def train(data_path, save_path, data_files, num_nodes, edge_index1, edge_index2, config):
    embed_size = config['embed_size']
    hidden_size = config['hidden_size']
    drop_rate = config['drop_rate']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    measure = config['loss_measure']
    is_weighted = config['weighted_loss']
    l_st = config['lambda_st']
    l_ss = l_tt = 0.5 * (1 - l_st)
    activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[config['activation']]

    graph_encoder1 = GraphEncoder(embed_size, hidden_size, GATConv, 2, activation)
    graph_encoder2 = GraphEncoder(embed_size, hidden_size, GATConv, 2, activation)
    seq_encoder = TransformerModel(hidden_size, 4, hidden_size, 2, drop_rate)
    model = MultiViewModel(num_nodes, embed_size, hidden_size, edge_index1, edge_index2,
                           graph_encoder1, graph_encoder2, seq_encoder).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_name = "_".join(['mv', str(config['lambda_st']), str(config['num_samples'])])
    checkpoints = [f for f in os.listdir(save_path) if f.startswith(model_name)]
    if not config['retrain'] and checkpoints:
        checkpoint_path = os.path.join(save_path, sorted(checkpoints)[-1])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
    else:
        model.apply(utils.weight_init)
        current_epoch = 1

    if current_epoch < num_epochs:
        data, w_rt = utils.train_data_loader(data_path, data_files, num_nodes, config)

        print("\n=== Training ===")
        for epoch in range(current_epoch, num_epochs + 1):
            for n, batch_index in enumerate(utils.next_batch_index(data.shape[0], batch_size)):
                data_batch = data[batch_index].cuda()
                w_batch = w_rt[batch_index].cuda() if w_rt is not None else 0
                model.train()
                optimizer.zero_grad()
                node_rep1, node_rep2, seq_rep1, seq_rep2 = model(data_batch)
                loss_ss = node_node_loss(node_rep1, node_rep2, measure)
                loss_tt = seq_seq_loss(seq_rep1, seq_rep2, measure)
                if is_weighted:
                    loss_st1 = weighted_ns_loss(node_rep1, seq_rep2, w_batch, measure)
                    loss_st2 = weighted_ns_loss(node_rep2, seq_rep1, w_batch, measure)
                else:
                    loss_st1 = node_seq_loss(node_rep1, seq_rep2, data_batch, measure)
                    loss_st2 = node_seq_loss(node_rep2, seq_rep1, data_batch, measure)
                loss_st = (loss_st1 + loss_st2) / 2
                loss = l_ss * loss_ss + l_tt * loss_tt + l_st * loss_st
                loss.backward()
                optimizer.step()
                if not (n + 1) % 200:
                    t = datetime.now().strftime('%m-%d %H:%M:%S')
                    print(f'{t} | (Train) | Epoch={epoch}, batch={n + 1} loss={loss.item():.4f}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_path, "_".join([model_name, f'{epoch}.pt'])))

    return model
