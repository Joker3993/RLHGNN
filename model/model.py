import pickle

import dgl
import dgl.nn.pytorch

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import LapPE
from dgl.nn.pytorch import LapPosEncoder, SAGEConv, GATConv, GatedGraphConv, GINConv, HeteroGraphConv, \
    GlobalAttentionPooling

import torch
import torch.nn.init as init


class HeteroSAGE(nn.Module):
    def __init__(self,
                 hidden_dim,
                 dropout,
                 dataname,
                 fold,
                 num_layers,
                 edge_types=[('node', 'forward', 'node'), ('node', 'backward', 'node'),
                             ('node', 'repeat_next', 'node')]):
        super(HeteroSAGE, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.edge_types = edge_types
        self.num_layers = num_layers

        features_name, vocab_sizes = self._load_feature_config(dataname, fold)
        self.max_length, self.mean_length = self._load_trace_config(dataname, fold)
        self.n_classes = self._load_class_config(dataname, fold) + 1

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(voca_size + 1, hidden_dim)
            for voca_size in vocab_sizes
        ])
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(features_name), hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_norm = nn.LayerNorm(hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

        self.hetero_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_dropouts = nn.ModuleList()

        aggregator_dict = {
            'forward': 'lstm',
            'backward': 'lstm',
            'repeat_next': 'mean'
        }

        for _ in range(num_layers):
            conv_dict = {
                etype: SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_dict[etype[1]])
                for etype in edge_types
            }
            self.hetero_convs.append(HeteroGraphConv(conv_dict, aggregate='mean'))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.residual_dropouts.append(nn.Dropout(dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_classes)
        )

        self._init_last_layer()
        self.init_weights()

    def _load_feature_config(self, dataname, fold):

        path = f"raw_dir/{dataname}_{fold}/"
        with open(path + "features_name.npy", 'rb') as f:
            features_name = pickle.load(f)
        vocab_sizes = [
            np.load(path + f"{feat}_info.npy", allow_pickle=True)
            for feat in features_name
        ]
        self.features_name = features_name
        return features_name, vocab_sizes

    def _load_trace_config(self, dataname, fold):

        path = f"raw_dir/{dataname}_{fold}/"
        with open(path + "max_trace.npy", 'rb') as f:
            max_len = pickle.load(f)
        with open(path + "mean_trace.npy", 'rb') as f:
            mean_len = pickle.load(f)
        return max_len, mean_len

    def _load_class_config(self, dataname, fold):

        path = f"raw_dir/{dataname}_{fold}/"
        return np.load(path + "activity_info.npy", allow_pickle=True)

    def _init_last_layer(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear) and module.out_features == self.n_classes:
                init.xavier_normal_(module.weight, gain=0.01)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == self.n_classes:
                    continue
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, hg):

        feat_embeds = []
        for feat_name, embed in zip(self.features_name, self.embedding_layers):
            feat_embeds.append(embed(hg.ndata[feat_name].long()))
        h = torch.cat(feat_embeds, dim=1)
        h = self.feature_proj(h)
        h = self.proj_norm(h)
        h = self.proj_dropout(h)

        for conv, norm, res_drop in zip(self.hetero_convs, self.norms, self.residual_dropouts):
            residual = h
            h_dict = conv(hg, {'node': h})
            h = h_dict['node']

            h = residual + res_drop(h - residual)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        with hg.local_scope():
            hg.ndata['h'] = h
            graph_embed = dgl.max_nodes(hg, 'h')
            return self.classifier(graph_embed)
