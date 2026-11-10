"""Figures for the Phase 2 (deep learning) IEEE report and slides -> report/dl_figures/.

Column-width figures are sized for a 3.5 in IEEE column; `wide_*` spans both columns.
Run:  .venv/bin/python report/make_dl_figures.py
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "dl_figures"
OUT.mkdir(exist_ok=True)

TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
         "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
ORDER = ["LogReg", "RandomForest", "XGBoost", "FP-MLP", "GCN", "GINE-MPNN", "AttentiveFP",
         "Hybrid", "Ensemble RF+GNN"]
SHORT = {"LogReg": "LR", "RandomForest": "RF", "XGBoost": "XGB", "XGBoost-FPonly": "XGB-FP",
         "FP-MLP": "FP-MLP", "GCN": "GCN", "GINE-MPNN": "GINE", "AttentiveFP": "AttFP",
         "Hybrid": "Hybrid", "Ensemble RF+GNN": "Ens."}
FAMILY = {"LogReg": "Classical ML", "RandomForest": "Classical ML", "XGBoost": "Classical ML",
          "XGBoost-FPonly": "Classical ML", "FP-MLP": "Deep (fingerprint)", "GCN": "Graph DL",
          "GINE-MPNN": "Graph DL", "AttentiveFP": "Graph DL", "Hybrid": "Hybrid", "Ensemble RF+GNN": "Hybrid"}
FC = {"Classical ML": "#2a78d6", "Deep (fingerprint)": "#eb6834", "Graph DL": "#1baf7a", "Hybrid": "#6250d6"}
INK, INK2, GRID = "#1b1b1a", "#52514e", "#dfe2e8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .5, "axes.axisbelow": True, "legend.frameon": False,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "savefig.transparent": False,
})
COL, WIDE = 3.45, 7.16  # inches

summ = pd.read_csv(RES / "summary_macro.csv")
allm = pd.read_csv(RES / "all_metrics.csv")
gap = pd.read_csv(RES / "scaffold_gap_by_endpoint.csv", index_col=0)
sim = pd.read_csv(RES / "auc_by_similarity.csv")


def s(split, model, col):
    r = summ[(summ.split == split) & (summ.model == model)]
    return np.nan if r.empty else float(r.iloc[0][col])


# --------------------------------------------------------------------------- 1. model comparison
fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5), sharey=True)
y = np.arange(len(ORDER))[::-1]
for ax, met, lab in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
    sc = [s("scaffold", m, met + "_mean") for m in ORDER]
    sd = [s("scaffold", m, met + "_std") for m in ORDER]
    rd = [s("random", m, met + "_mean") for m in ORDER]
    ax.barh(y + .19, rd, .36, color=[FC[FAMILY[m]] for m in ORDER], alpha=.35, label="random split")
    ax.barh(y - .19, sc, .36, xerr=sd, color=[FC[FAMILY[m]] for m in ORDER],
            error_kw={"ecolor": INK2, "lw": .7, "capsize": 1.6}, label="scaffold split")
    for yi, v in zip(y, sc):
        ax.text(v + .006, yi - .19, f"{v:.3f}", va="center", fontsize=6.2, color=INK)
    lo = min(np.nanmin(sc), np.nanmin(rd)) - .055
    ax.set_xlim(lo, max(np.nanmax(sc), np.nanmax(rd)) + .035)
    ax.set_xlabel(f"Mean {lab} over 12 endpoints")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, [SHORT[m] for m in ORDER])
h = [mpatches.Patch(color=FC[f], label=f) for f in ["Classical ML", "Deep (fingerprint)", "Graph DL", "Hybrid"]]
h += [mpatches.Patch(facecolor="#777", alpha=.35, label="random split"),
      mpatches.Patch(facecolor="#777", label="scaffold split")]
fig.legend(handles=h, loc="upper center", ncol=6, bbox_to_anchor=(.5, 1.10), handlelength=1.2, columnspacing=1.1)
fig.tight_layout()
fig.savefig(OUT / "wide_models.png")
plt.close(fig)

# --------------------------------------------------------------------------- 2. learning curves
# Hybrid was re-trained after the fix in gnn.Hybrid: its re-run files supersede the originals.
base = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(RES / "gnn_history_seed*.csv")))])
rerun = [pd.read_csv(p) for p in sorted(glob.glob(str(RES / "gnn_history_hybrid*.csv")))]
hist = pd.concat([base[base.model != "Hybrid"]] + rerun) if rerun else base
hist = hist[(hist.split == "scaffold") & (hist.seed == 0)]
fig, axes = plt.subplots(1, 2, figsize=(COL * 2 + .26, 2.0))
styles = {"FP-MLP": ("-", "#eb6834"), "GCN": ("-", "#1baf7a"), "GINE-MPNN": ("--", "#15805a"),
          "AttentiveFP": (":", "#0f9f72"), "Hybrid": ("-", "#6250d6")}
for mdl, g in hist.groupby("model"):
    g = g.sort_values("epoch")
    ls, c = styles.get(mdl, ("-", "#888"))
    axes[0].plot(g.epoch, g.train_loss, ls, color=c, lw=1.1, label=mdl)
    axes[1].plot(g.epoch, g.val_roc_auc, ls, color=c, lw=1.1, label=mdl)
axes[0].set_ylabel("Masked weighted BCE (train)")
axes[1].set_ylabel("Mean validation ROC-AUC")
axes[1].set_ylim(.62, .85)
for ax in axes:
    ax.set_xlabel("Epoch")
axes[1].legend(ncol=2, loc="lower right", handlelength=1.5)
fig.tight_layout()
fig.savefig(OUT / "wide_curves.png")
plt.close(fig)

# --------------------------------------------------------------------------- 3. generalisation gap
fig, ax = plt.subplots(figsize=(COL, 1.85))
mg = gap.mean(axis=1).reindex(ORDER).dropna()
ax.bar(range(len(mg)), mg.values, .6, color=[FC[FAMILY[m]] for m in mg.index])
for i, v in enumerate(mg.values):
    ax.text(i, v + .0012, f"{v:.3f}", ha="center", fontsize=6, color=INK)
ax.set_xticks(range(len(mg)), [SHORT[m] for m in mg.index], rotation=35, ha="right")
ax.set_ylim(0, mg.max() * 1.22)
ax.set_ylabel("ROC-AUC(random) − ROC-AUC(scaffold)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "col_gap.png")
plt.close(fig)

# --------------------------------------------------------------------------- 4. AUC vs similarity
fig, ax = plt.subplots(figsize=(COL, 2.0))
bins = ["<0.3", "0.3–0.4", "0.4–0.5", "0.5–0.7", "≥0.7"]
show = ["RandomForest", "FP-MLP", "GCN", "AttentiveFP", "Ensemble RF+GNN"]
mk = ["o", "s", "^", "v", "D"]
for m, k in zip(show, mk):
    d = sim[sim.model == m].set_index("bin").reindex(bins)
    ax.plot(bins, d.mean_roc_auc, marker=k, ms=3.2, lw=1.1, color=FC[FAMILY[m]], label=SHORT[m],
            ls="-" if FAMILY[m] != "Graph DL" else "--")
n = sim.drop_duplicates("bin").set_index("bin").reindex(bins).n_molecules
ax.set_xticks(range(5), [f"{b}\n$n$={v}" for b, v in zip(bins, n)])
ax.set_xlabel("Tanimoto similarity to nearest training molecule")
ax.set_ylabel("Mean ROC-AUC")
ax.legend(ncol=2, handlelength=1.6, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "col_similarity.png")
plt.close(fig)

# --------------------------------------------------------------------------- 5. endpoint heatmap
hm = (allm[allm.split == "scaffold"].groupby(["model", "endpoint"]).roc_auc.mean().unstack()[TASKS]
      .reindex(ORDER).dropna(how="all"))
fig, ax = plt.subplots(figsize=(WIDE, 2.25))
im = ax.imshow(hm.values, cmap="Blues", aspect="auto")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        sh = (v - hm.values.min()) / (hm.values.max() - hm.values.min())
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.9,
                color="white" if sh > .6 else INK,
                fontweight="bold" if v == hm.values[:, j].max() else "normal")
ax.set_xticks(range(12), TASKS, rotation=30, ha="right", fontsize=6.4)
ax.set_yticks(range(len(hm)), [SHORT[m] for m in hm.index], fontsize=6.6)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=.016, pad=.008).set_label("ROC-AUC", size=6.6)
fig.tight_layout()
fig.savefig(OUT / "wide_heatmap.png")
plt.close(fig)

# --------------------------------------------------------------------------- 6. architecture schematic
fig, ax = plt.subplots(figsize=(WIDE, 1.95))
ax.set_xlim(0, 100)
ax.set_ylim(0, 24)
ax.axis("off")
ax.grid(False)


def box(x, w, label, sub, color, y=3.5, h=15.0, subsize=6.2):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                         fc="white", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h - 2.8, label, ha="center", va="center", fontsize=7.4, color=INK, weight="bold")
    ax.text(x + w / 2, y + (h - 4.6) / 2, sub, ha="center", va="center", fontsize=subsize,
            color=INK2, linespacing=1.6)


def flow(x1, x2, y=11.0):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))


# molecule graph
ax.add_patch(mpatches.FancyBboxPatch((0.4, 3.5), 16.4, 15.0, boxstyle="round,pad=0.3,rounding_size=1.0",
                                     fc="white", ec="#2a78d6", lw=1.3))
ax.text(8.6, 16.3, "Molecular graph", ha="center", va="center", fontsize=7.4, color=INK, weight="bold")
gx, gy = 8.6, 11.3
pos = [(-3.3, -1.4), (-1.1, 1.0), (2.1, 0.5), (3.0, -2.4), (0.8, -4.0), (-2.4, -3.5), (-3.0, 2.9)]
bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6)]
for a, b in bonds:
    ax.plot([gx + pos[a][0], gx + pos[b][0]], [gy + pos[a][1], gy + pos[b][1]], "-",
            color="#8a909c", lw=1.0, zorder=1)
for i, (px, py) in enumerate(pos):
    ax.add_patch(mpatches.Ellipse((gx + px, gy + py), 1.5, 1.05, fc="#eb6834" if i == 1 else "#2a78d6",
                                  ec="white", lw=.8, zorder=2))
ax.text(8.6, 5.4, "atoms = nodes (56-d)\nbonds = edges (12-d)", ha="center", va="center",
        fontsize=6.1, color=INK2, linespacing=1.6)
flow(17.2, 21.4)
box(21.9, 24.5, "Message passing $\\times\\,k$",
    "$h_v^{(k+1)}=U\\!\\left(h_v^{(k)},\\;\\mathrm{AGG}_{u\\in N(v)}\\,M(h_u^{(k)},e_{uv})\\right)$\n\n"
    "GCN  ·  GINE  ·  AttentiveFP", "#1baf7a", subsize=5.9)
flow(46.9, 51.1)
box(51.6, 15.5, "Readout", "sum ‖ mean pooling\nover all atoms\n$\\rightarrow$ molecule vector", "#1baf7a")
flow(67.6, 71.8)
box(72.3, 13.5, "MLP head", "256 $\\rightarrow$ 12 logits\n(one per endpoint)", "#6250d6")
flow(86.1, 90.3)
ax.text(95.4, 20.0, "12 endpoints", ha="center", fontsize=6.9, color=INK, weight="bold")
lab = [1, 0, None, 1, 0, 0, None, 1, 0, 0, 1, None]
for i, v in enumerate(lab):
    yy = 17.6 - i * 0.98
    c = {1: "#e34948", 0: "#1baf7a", None: "#d7dbe2"}[v]
    ax.add_patch(mpatches.Rectangle((91.6, yy), 7.6, .72, fc=c, ec="none"))
ax.text(95.4, 4.0, "masked loss: grey =\nlabel missing, so it\ncontributes no gradient",
        ha="center", va="center", fontsize=6.1, color=INK2, linespacing=1.6)
fig.tight_layout(pad=.12)
fig.savefig(OUT / "wide_architecture.png")
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))

# --------------------------------------------------------------------------- 7. column-width variants
# Used in the IEEE report so that only two figures need a full-width (spanning) frame.
fig, axes = plt.subplots(2, 1, figsize=(COL, 4.35))
for ax, met, lab in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
    sc = [s("scaffold", m, met + "_mean") for m in ORDER]
    sd = [s("scaffold", m, met + "_std") for m in ORDER]
    rd = [s("random", m, met + "_mean") for m in ORDER]
    ax.barh(y + .19, rd, .36, color=[FC[FAMILY[m]] for m in ORDER], alpha=.33)
    ax.barh(y - .19, sc, .36, xerr=sd, color=[FC[FAMILY[m]] for m in ORDER],
            error_kw={"ecolor": INK2, "lw": .6, "capsize": 1.4})
    for yi, v in zip(y, sc):
        ax.text(v + .005, yi - .19, f"{v:.3f}", va="center", fontsize=5.8, color=INK)
    lo = min(np.nanmin(sc), np.nanmin(rd)) - .05
    ax.set_xlim(lo, max(np.nanmax(sc), np.nanmax(rd)) + .04)
    ax.set_yticks(y, [SHORT[m] for m in ORDER])
    ax.set_xlabel(f"Mean {lab} over 12 endpoints")
    ax.grid(axis="y", visible=False)
h = [mpatches.Patch(color=FC[f], label=f) for f in ["Classical ML", "Deep (fingerprint)", "Graph DL", "Hybrid"]]
h += [mpatches.Patch(facecolor="#777", alpha=.33, label="random split"),
      mpatches.Patch(facecolor="#777", label="scaffold split")]
fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.10), handlelength=1.1,
           columnspacing=1.0, fontsize=6.2)
fig.tight_layout()
fig.savefig(OUT / "col_models.png")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(COL, 3.5))
for mdl, grp in hist.groupby("model"):
    grp = grp.sort_values("epoch")
    ls, c = styles.get(mdl, ("-", "#888"))
    axes[0].plot(grp.epoch, grp.train_loss, ls, color=c, lw=1.0, label=mdl)
    axes[1].plot(grp.epoch, grp.val_roc_auc, ls, color=c, lw=1.0, label=mdl)
axes[0].set_ylabel("Masked weighted BCE (train)")
axes[1].set_ylabel("Mean validation ROC-AUC")
axes[1].set_ylim(.62, .85)
axes[1].set_xlabel("Epoch")
axes[0].legend(ncol=2, handlelength=1.4, fontsize=6.0, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "col_curves.png")
plt.close(fig)
print("column variants done")
