"""DL models: multitask networks trained with a masked loss for missing labels.

Models
------
* FP-MLP      : multitask MLP on the *same* engineered features as the AML
                baselines. Control that separates the effect of multitask deep
                learning from the effect of learned graph representations.
* GCN         : graph convolutional network (Kipf & Welling) on atom features.
* GINE-MPNN   : message passing network (GIN with edge features), messages are
                conditioned on bond type / conjugation / ring / stereo.
* AttentiveFP : attention-based message passing (Xiong et al., 2019).
* Hybrid      : GINE-MPNN graph embedding concatenated with the engineered
                fingerprint/descriptor vector before the prediction head.

All models output 12 logits (one per endpoint). The loss is binary cross-entropy
evaluated only where a label is observed; per-endpoint positive-class weights
(#neg / #pos on the training labels of that endpoint) counter class imbalance.
Early stopping uses mean validation ROC-AUC.

Run:  python -m src.gnn [--models GCN GINE-MPNN ...] [--splits scaffold random]
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (AttentiveFP, GCNConv, GINEConv, global_add_pool,
                                global_max_pool, global_mean_pool)

from .config import RESULTS, SPLIT_SEEDS, SPLIT_TYPES, TASKS
from .data import get_split, label_matrix, load_tox21
from .features import (N_ATOM_FEATURES, N_BOND_FEATURES, DescriptorScaler,
                       load_engineered, load_graphs)
from .metrics import endpoint_metrics, mean_roc_auc

PRED_DIR = RESULTS / "preds"
PRED_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")
N_TASKS = len(TASKS)

HPARAMS = dict(hidden=256, layers=4, dropout=0.2, lr=1e-3, weight_decay=1e-5,
               batch_size=128, max_epochs=100, patience=20)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def head(in_dim, hidden, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, N_TASKS))


class FPMLP(nn.Module):
    def __init__(self, feat_dim, hidden, dropout, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_TASKS))

    def forward(self, batch, feats):
        return self.net(feats[batch.idx])


class GCN(nn.Module):
    def __init__(self, hidden, layers, dropout, **_):
        super().__init__()
        self.inp = nn.Linear(N_ATOM_FEATURES, hidden)
        self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = dropout
        self.out = head(2 * hidden, hidden, dropout)

    def embed(self, b):
        h = self.inp(b.x)
        for conv, bn in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(bn(conv(h, b.edge_index))), self.dropout, self.training)
        return torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], 1)

    def forward(self, b, feats=None):
        return self.out(self.embed(b))


class GINEMPNN(nn.Module):
    def __init__(self, hidden, layers, dropout, **_):
        super().__init__()
        self.inp = nn.Linear(N_ATOM_FEATURES, hidden)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden)), edge_dim=N_BOND_FEATURES)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = dropout
        self.out_dim = 2 * hidden
        self.out = head(self.out_dim, hidden, dropout)

    def embed(self, b):
        h = self.inp(b.x)
        for conv, bn in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(bn(conv(h, b.edge_index, b.edge_attr))), self.dropout, self.training)
        return torch.cat([global_add_pool(h, b.batch), global_mean_pool(h, b.batch)], 1)

    def forward(self, b, feats=None):
        return self.out(self.embed(b))


class AttentiveFPNet(nn.Module):
    def __init__(self, hidden, dropout, **_):
        super().__init__()
        self.net = AttentiveFP(N_ATOM_FEATURES, hidden, N_TASKS, edge_dim=N_BOND_FEATURES,
                               num_layers=3, num_timesteps=2, dropout=dropout)

    def forward(self, b, feats=None):
        return self.net(b.x, b.edge_index, b.edge_attr, b.batch)


class Hybrid(nn.Module):
    """Learned graph embedding + engineered fingerprint/descriptor vector.

    The fingerprint branch fits the training set within a few epochs while the
    graph branch needs tens of epochs, so a naive concatenation early-stops
    before the graph side has learned anything. Two remedies: (i) heavy input
    dropout and a narrow projection on the fingerprint branch, and (ii)
    "modality dropout", which zeroes the whole fingerprint embedding for a
    random half of the training molecules so the head must also rely on the
    graph embedding.
    """

    def __init__(self, feat_dim, hidden, layers, dropout, fp_dim=128, p_modality=0.5, **_):
        super().__init__()
        self.gnn = GINEMPNN(hidden, layers, dropout)
        self.fp = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, 256), nn.BatchNorm1d(256),
                                nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, fp_dim), nn.ReLU())
        self.p_modality = p_modality
        self.out = head(self.gnn.out_dim + fp_dim, hidden, dropout)

    def forward(self, b, feats):
        h_fp = self.fp(feats[b.idx])
        if self.training:
            keep = (torch.rand(h_fp.size(0), 1, device=h_fp.device) > self.p_modality).float()
            h_fp = h_fp * keep / (1 - self.p_modality)
        return self.out(torch.cat([self.gnn.embed(b), h_fp], 1))


MODELS = {"FP-MLP": FPMLP, "GCN": GCN, "GINE-MPNN": GINEMPNN,
          "AttentiveFP": AttentiveFPNet, "Hybrid": Hybrid}
FAMILY = {"FP-MLP": "Fingerprint DL", "GCN": "Graph DL", "GINE-MPNN": "Graph DL",
          "AttentiveFP": "Graph DL", "Hybrid": "Hybrid (graph + FP)"}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
