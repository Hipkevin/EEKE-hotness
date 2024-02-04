import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.word_emb_dim, args.word_emb_dim)
        self.conv2 = GCNConv(args.word_emb_dim, args.word_emb_dim)

        self.linear = nn.Linear(args.word_emb_dim, 1)

        self.dropout = nn.Dropout()

    def forward(self, graph):
        x = self.conv1(graph.w_x, graph.edge_index, graph.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, graph.edge_index, graph.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear(x)

        return x

class StaticAttModel(nn.Module):
    def __init__(self, args):
        super(StaticAttModel, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=args.word_emb_dim,
                                               num_heads=args.att_head_num, batch_first=True)

        self.time_linear = nn.Sequential(
            nn.Linear(args.window_size, args.window_size),
            nn.ReLU(),
            nn.Linear(args.window_size, 1)
        )
        self.linear = nn.Linear(args.word_emb_dim, 1)

    def forward(self, graph):
        # (Q, K, V) ==> (S, W, W)
        # attn_output, attn_output_weights
        mixed_feature, _ = self.multi_att(query=graph.s_x,
                                       key=graph.w_x,
                                       value=graph.w_x)

        time_feature = self.time_linear(graph.y_memory)

        y_hat = self.linear(mixed_feature) + time_feature

        return y_hat

class StaticCatModel(nn.Module):
    def __init__(self, args):
        super(StaticCatModel, self).__init__()

        self.time_linear = nn.Sequential(
            nn.Linear(args.window_size, args.window_size),
            nn.ReLU(),
            nn.Linear(args.window_size, 1)
        )
        self.linear = nn.Linear(2 * args.word_emb_dim, 1)

    def forward(self, graph):
        # (words_num, dim) & (words_num, dim) ==> (words_num, 2 * dim)
        mixed_feature = torch.cat([graph.s_x, graph.w_x], dim=-1)

        time_feature = self.time_linear(graph.y_memory)

        y_hat = self.linear(mixed_feature) + time_feature

        return y_hat