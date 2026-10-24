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
def masked_bce(logits, y, pos_weight):
    """BCE over observed labels only; NaN labels contribute nothing."""
    mask = ~torch.isnan(y)
    y0 = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, y0, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict(model, loader, feats):
    model.eval()
    out = [torch.sigmoid(model(b.to(DEVICE), feats)).cpu() for b in loader]
    return torch.cat(out).numpy()


def train_one(name, graphs, feats, y, tr, va, te, seed, hp=HPARAMS, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ytr = y[tr]
    n_pos = np.nansum(ytr, 0)
    n_obs = (~np.isnan(ytr)).sum(0)
    pos_weight = torch.tensor((n_obs - n_pos) / np.maximum(n_pos, 1), dtype=torch.float, device=DEVICE)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader([graphs[i] for i in tr], batch_size=hp["batch_size"],
                              shuffle=True, generator=g, drop_last=True)
    valid_loader = DataLoader([graphs[i] for i in va], batch_size=512)
    test_loader = DataLoader([graphs[i] for i in te], batch_size=512)

    model = MODELS[name](feat_dim=feats.shape[1], **hp).to(DEVICE)
    if name == "Hybrid":  # stronger L2 on the fast-overfitting fingerprint branch
        fp_params = list(model.fp.parameters())
        fp_ids = {id(p) for p in fp_params}
        groups = [{"params": [p for p in model.parameters() if id(p) not in fp_ids]},
                  {"params": fp_params, "weight_decay": 1e-2}]
    else:
        groups = model.parameters()
    opt = torch.optim.AdamW(groups, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)

    best_auc, best_state, best_epoch, history = -1, None, 0, []
    for epoch in range(hp["max_epochs"]):
        model.train()
        tot = 0.0
        for b in train_loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = masked_bce(model(b, feats), b.y, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        val_auc = mean_roc_auc(y[va], predict(model, valid_loader, feats))
        sched.step(val_auc)
        history.append({"epoch": epoch, "train_loss": tot / len(train_loader), "val_roc_auc": val_auc})
        if val_auc > best_auc:
            best_auc, best_epoch, best_state = val_auc, epoch, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {epoch:3d} loss={tot/len(train_loader):.4f} val_auc={val_auc:.4f}", flush=True)
        if epoch - best_epoch >= hp["patience"]:
            break

    model.load_state_dict(best_state)
    return (model, predict(model, valid_loader, feats), predict(model, test_loader, feats),
            best_auc, best_epoch, history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--splits", nargs="+", default=SPLIT_TYPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SPLIT_SEEDS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global DEVICE
    DEVICE = torch.device(args.device)

    df = load_tox21()
    y, _ = label_matrix(df)
    graphs = load_graphs(df, y)
    for i, gr in enumerate(graphs):
        gr.idx = torch.tensor([i])
    fp, desc, _ = load_engineered(df)

    out_csv = RESULTS / f"gnn_metrics{args.tag}.csv"
    hist_csv = RESULTS / f"gnn_history{args.tag}.csv"
    rows, hists = [], []
    for split_type in args.splits:
        for seed in args.seeds:
            tr, va, te = get_split(df, split_type, seed)
            scaler = DescriptorScaler().fit(desc[tr])
            feats = torch.tensor(np.hstack([fp.astype(np.float32), scaler.transform(desc)])).to(DEVICE)
            for name in args.models:
                t0 = time.time()
                _, p_va, p_te, val_auc, ep, hist = train_one(name, graphs, feats, y, tr, va, te, seed)
                np.savez(PRED_DIR / f"{split_type}_{seed}_{name}.npz", test_idx=te, valid_idx=va,
                         test=p_te, valid=p_va)
                for r in endpoint_metrics(y[te], p_te):
                    rows.append({"model": name, "family": FAMILY[name], "split": split_type,
                                 "seed": seed, **r})
                for h in hist:
                    hists.append({"model": name, "split": split_type, "seed": seed, **h})
                test_auc = mean_roc_auc(y[te], p_te)
                print(f"  {split_type}/{seed} {name:12s} best_epoch={ep:3d} val={val_auc:.4f} "
                      f"test={test_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                pd.DataFrame(hists).to_csv(hist_csv, index=False)


if __name__ == "__main__":
    main()
