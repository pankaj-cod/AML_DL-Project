"""Aggregate results, run statistics and produce every figure / table of the report.

Run:  python -m src.analysis
"""
import json
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from .config import DATA_PROCESSED, FIGURES, RESULTS, SPLIT_SEEDS, TASKS
from .data import get_split, label_matrix, load_tox21
from .features import morgan_fp
from .gnn import FAMILY

matplotlib.use("Agg")

MODEL_ORDER = ["LogReg", "RandomForest", "XGBoost", "XGBoost-FPonly", "FP-MLP",
               "GCN", "GINE-MPNN", "AttentiveFP", "Hybrid", "Ensemble RF+GNN"]
FAMILY_ORDER = ["Fingerprint AML", "Fingerprint DL", "Graph DL", "Hybrid (graph + FP)"]
# Validated categorical palette (slots 1-4), one colour per model family.
FAMILY_COLOR = dict(zip(FAMILY_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load_metrics():
    frames = [pd.read_csv(RESULTS / "baseline_metrics.csv")]
    frames += [pd.read_csv(p) for p in sorted(RESULTS.glob("gnn_metrics_seed*.csv"))]
    rerun = sorted(RESULTS.glob("gnn_metrics_hybrid*.csv"))  # Hybrid re-trained after the fix in gnn.Hybrid
    if rerun:
        frames = [f[f["model"] != "Hybrid"] for f in frames] + [pd.read_csv(p) for p in rerun]
    if (RESULTS / "ensemble_metrics.csv").exists():
        frames.append(pd.read_csv(RESULTS / "ensemble_metrics.csv"))
    m = pd.concat(frames, ignore_index=True)
    m = m[m["model"].isin(MODEL_ORDER)]
    m.to_csv(RESULTS / "all_metrics.csv", index=False)
    return m


def models_present(m):
    return [x for x in MODEL_ORDER if x in set(m["model"])]


def family_of(m):
    return m.drop_duplicates("model").set_index("model")["family"].to_dict()


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def summary_table(m):
    """Macro (mean over endpoints) metric per seed, then mean ± std over seeds."""
    per_seed = m.groupby(["split", "model", "family", "seed"])[["roc_auc", "pr_auc"]].mean().reset_index()
    s = per_seed.groupby(["split", "model", "family"]).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std")).reset_index()
    s["model"] = pd.Categorical(s["model"], MODEL_ORDER, ordered=True)
    s = s.sort_values(["split", "model"])
    s.to_csv(RESULTS / "summary_macro.csv", index=False)
    return s


def endpoint_table(m, metric, split="scaffold"):
    t = (m[m["split"] == split].groupby(["model", "endpoint"])[metric].agg(["mean", "std"])
         .reset_index())
    mean = t.pivot(index="model", columns="endpoint", values="mean").reindex(models_present(m))[TASKS]
    std = t.pivot(index="model", columns="endpoint", values="std").reindex(models_present(m))[TASKS]
    mean = mean.dropna(how="all")
    std = std.loc[mean.index]
    mean.to_csv(RESULTS / f"endpoint_{metric}_{split}_mean.csv")
    std.to_csv(RESULTS / f"endpoint_{metric}_{split}_std.csv")
    return mean, std


def significance_tests(m, split="scaffold"):
    """Paired Wilcoxon signed-rank tests over (endpoint, seed) pairs."""
    d = m[m["split"] == split].pivot_table(index=["endpoint", "seed"], columns="model", values="roc_auc")
    models = [x for x in MODEL_ORDER if x in d.columns]
    rows = []
    for a, b in combinations(models, 2):
        x = d[[a, b]].dropna()
        diff = x[a] - x[b]
        p = wilcoxon(x[a], x[b]).pvalue if (diff != 0).any() else 1.0
        rows.append({"model_a": a, "model_b": b, "mean_diff_roc_auc": diff.mean(),
                     "a_wins": int((diff > 0).sum()), "b_wins": int((diff < 0).sum()),
                     "n_pairs": len(x), "p_value": p})
    out = pd.DataFrame(rows)
    out["significant_0.05"] = out["p_value"] < 0.05
    out.to_csv(RESULTS / f"wilcoxon_{split}.csv", index=False)
    return out


def generalisation_gap(m):
    """Random-split minus scaffold-split ROC-AUC, per model and endpoint."""
    t = m.groupby(["model", "split", "endpoint"])["roc_auc"].mean().unstack("split").dropna()
    t["gap"] = t["random"] - t["scaffold"]
    gap = t["gap"].unstack("endpoint").reindex([x for x in MODEL_ORDER if x in t.index.get_level_values(0)])[TASKS]
    gap.to_csv(RESULTS / "scaffold_gap_by_endpoint.csv")
    return gap


# --------------------------------------------------------------------------- #
# Similarity of test molecules to the training set
# --------------------------------------------------------------------------- #
def nearest_train_similarity(df, bitvecs, split_type, seed):
    tr, _, te = get_split(df, split_type, seed)
    train_fps = [bitvecs[i] for i in tr]
    return te, np.array([max(DataStructs.BulkTanimotoSimilarity(bitvecs[i], train_fps)) for i in te])


def similarity_analysis(df, models):
    """ROC-AUC as a function of a test molecule's similarity to its nearest
    training neighbour, pooling the test sets of every split and seed."""
    cache = DATA_PROCESSED / "test_similarity.csv"
    y, _ = label_matrix(df)
    _, bitvecs = morgan_fp(df["smiles"])
    if cache.exists():
        sim = pd.read_csv(cache)
    else:
        recs = []
        for st in ["scaffold", "random"]:
            for seed in SPLIT_SEEDS:
                te, s = nearest_train_similarity(df, bitvecs, st, seed)
                recs += [{"split": st, "seed": seed, "idx": i, "sim": v} for i, v in zip(te, s)]
        sim = pd.DataFrame(recs)
        sim.to_csv(cache, index=False)

    bins = [0, 0.3, 0.4, 0.5, 0.7, 1.01]
    labels = ["<0.3", "0.3–0.4", "0.4–0.5", "0.5–0.7", "≥0.7"]
    sim["bin"] = pd.cut(sim["sim"], bins, labels=labels, right=False)
    rows = []
    for model in models:
        parts = []
        for (st, seed), g in sim.groupby(["split", "seed"]):
            f = RESULTS / "preds" / f"{st}_{seed}_{model}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            pos = {v: k for k, v in enumerate(z["test_idx"])}
            p = z["test"][[pos[i] for i in g["idx"]]]
            parts.append(pd.DataFrame(p, columns=TASKS).assign(bin=g["bin"].values, idx=g["idx"].values))
        if not parts:
            continue
        P = pd.concat(parts, ignore_index=True)
        for b, g in P.groupby("bin", observed=True):
            aucs = []
            for t, task in enumerate(TASKS):
                yt = y[g["idx"].values, t]
                ok = ~np.isnan(yt)
                if ok.sum() > 0 and 0 < yt[ok].sum() < ok.sum() and yt[ok].sum() >= 3:
                    aucs.append(roc_auc_score(yt[ok], g[task].values[ok]))
            rows.append({"model": model, "bin": b, "n_molecules": len(g), "mean_roc_auc": np.mean(aucs),
                         "n_endpoints": len(aucs)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "auc_by_similarity.csv", index=False)
    return sim, out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_dataset(df):
    y, mask = label_matrix(df)
    prev = np.nansum(y, 0) / mask.sum(0)
    missing = 1 - mask.mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(TASKS, prev * 100, color=FAMILY_COLOR["Fingerprint AML"], width=0.6)
    for i, v in enumerate(prev * 100):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    axes[0].set_ylabel("Active compounds (%)")
    axes[0].set_title("Positive rate per endpoint (severe imbalance)", loc="left")
    axes[1].bar(TASKS, missing * 100, color=INK2, width=0.6)
    axes[1].set_ylabel("Missing labels (%)")
    axes[1].set_title("Missing labels per endpoint (handled by masking)", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_dataset.png")
    plt.close(fig)


def fig_scaffolds(df, sim):
    sizes = df["scaffold"].fillna("").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.hist(sizes.values, bins=np.logspace(0, np.log10(sizes.max() + 1), 30),
            color=FAMILY_COLOR["Fingerprint AML"], edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Molecules per Bemis–Murcko scaffold")
    ax.set_ylabel("Number of scaffolds")
    ax.set_title(f"{len(sizes)} scaffolds; largest = acyclic ({sizes.iloc[0]}) "
                 f"and benzene ({sizes.iloc[1]})", loc="left", fontsize=9)
    ax = axes[1]
    for st, c in [("random", INK2), ("scaffold", FAMILY_COLOR["Hybrid (graph + FP)"])]:
        v = sim.loc[sim["split"] == st, "sim"]
        ax.hist(v, bins=np.linspace(0, 1, 41), alpha=0.65, color=c, edgecolor="white",
                label=f"{st} split (median {v.median():.2f})")
    ax.set_xlabel("Tanimoto similarity of test molecule to nearest training molecule")
    ax.set_ylabel("Test molecules")
    ax.set_title("Scaffold split removes near-duplicate neighbours", loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_scaffold_split.png")
    plt.close(fig)


def fig_heatmap(mean, std, metric, title, fname, prevalence=None):
    data = mean.copy()
    if prevalence is not None:
        data = pd.concat([pd.DataFrame([prevalence], index=["(random baseline)"]), data])
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(data) + 1.6))
    im = ax.imshow(data.values, cmap="Blues", aspect="auto",
                   vmin=np.nanmin(data.values), vmax=np.nanmax(data.values))
    for i in range(data.shape[0]):
        col_best = None
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if np.isnan(v):
                continue
            best = (prevalence is None or i > 0) and v == np.nanmax(mean.values[:, j])
            shade = (v - np.nanmin(data.values)) / (np.nanmax(data.values) - np.nanmin(data.values))
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if shade > 0.6 else INK, fontweight="bold" if best else "normal")
    ax.set_xticks(range(len(TASKS)), TASKS, rotation=45, ha="right")
    ax.set_yticks(range(len(data)), data.index)
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=metric)
    fig.tight_layout()
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def fig_summary(s, fam):
    models = [x for x in MODEL_ORDER if x in set(s["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in [(axes[0], "roc_auc", "Mean ROC-AUC"), (axes[1], "pr_auc", "Mean PR-AUC")]:
        x = np.arange(len(models))
        for k, (st, hatch, alpha) in enumerate([("scaffold", None, 1.0), ("random", "///", 0.45)]):
            d = s[s["split"] == st].set_index("model").reindex(models)
            ax.bar(x + (k - 0.5) * 0.38, d[f"{metric}_mean"], 0.36, yerr=d[f"{metric}_std"],
                   color=[FAMILY_COLOR[fam[mm]] for mm in models], alpha=alpha, hatch=hatch,
                   edgecolor="white", linewidth=1.5, capsize=2, error_kw={"ecolor": INK2, "lw": 0.8},
                   label=f"{st} split")
        lo = np.nanmin(s[f"{metric}_mean"]) - 0.05
        ax.set_ylim(lo, np.nanmax(s[f"{metric}_mean"]) + 0.03)
        ax.set_xticks(x, models, rotation=40, ha="right")
        ax.set_ylabel(label + " (12 endpoints, 3 seeds)")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_ORDER]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2),
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="///")]
    fig.legend(handles, FAMILY_ORDER + ["solid = scaffold split", "hatched = random split"],
               loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_summary_scaffold_vs_random.png")
    plt.close(fig)


def fig_gap(gap, fam):
    mean_gap = gap.mean(1)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(mean_gap.index, mean_gap.values, color=[FAMILY_COLOR[fam[m]] for m in mean_gap.index], width=0.6)
    for i, v in enumerate(mean_gap.values):
        ax.text(i, v + 0.002, f"{v:+.3f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(0, mean_gap.max() * 1.2)
    ax.set_ylabel("ROC-AUC(random) − ROC-AUC(scaffold)")
    ax.set_title("Scaffold generalisation gap (lower = generalises better to new chemotypes)", loc="left")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_generalisation_gap.png")
    plt.close(fig)


def fig_similarity(sim_auc, fam, show):
    fig, ax = plt.subplots(figsize=(8, 4))
    markers = ["o", "s", "^", "D", "v", "P"]
    for k, m in enumerate(show):
        d = sim_auc[sim_auc["model"] == m]
        ax.plot(d["bin"].astype(str), d["mean_roc_auc"], marker=markers[k % 6], ms=7, lw=2,
                color=FAMILY_COLOR[fam[m]], label=m,
                linestyle="-" if k % 2 == 0 else "--")
    counts = sim_auc.drop_duplicates("bin")[["bin", "n_molecules"]]
    ax.set_xticks(range(len(counts)),
                  [f"{b}\n(n={n})" for b, n in zip(counts["bin"].astype(str), counts["n_molecules"])])
    ax.set_xlabel("Tanimoto similarity to nearest training molecule (pooled test sets, all splits)")
    ax.set_ylabel("Mean ROC-AUC over endpoints")
    ax.set_title("Accuracy decays for chemistry far from the training set", loc="left")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_auc_vs_similarity.png")
    plt.close(fig)


def fig_learning_curves():
    paths = sorted(RESULTS.glob("gnn_history_seed*.csv"))
    if not paths:
        return
    h = pd.concat([pd.read_csv(p) for p in paths])
    rerun = sorted(RESULTS.glob("gnn_history_hybrid*.csv"))
    if rerun:
        h = pd.concat([h[h["model"] != "Hybrid"]] + [pd.read_csv(p) for p in rerun])
    h = h[(h["split"] == "scaffold") & (h["seed"] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (m, g) in enumerate(h.groupby("model", sort=False)):
        ls = ["-", "--", ":", "-.", "-"][k % 5]
        axes[0].plot(g["epoch"], g["train_loss"], ls, color=FAMILY_COLOR[FAMILY[m]], label=m, lw=1.6)
        axes[1].plot(g["epoch"], g["val_roc_auc"], ls, color=FAMILY_COLOR[FAMILY[m]], label=m, lw=1.6)
    axes[0].set_ylabel("Masked weighted BCE (train)")
    axes[1].set_ylabel("Mean validation ROC-AUC")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[1].legend(ncol=2, fontsize=8)
    axes[0].set_title("Training curves, scaffold split seed 0", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_learning_curves.png")
    plt.close(fig)


def main():
    df = load_tox21()
    m = load_metrics()
    fam = family_of(m)
    s = summary_table(m)
    roc_mean, roc_std = endpoint_table(m, "roc_auc")
    pr_mean, pr_std = endpoint_table(m, "pr_auc")
    prevalence = (m[(m["split"] == "scaffold")].groupby("endpoint")["prevalence"].mean()[TASKS])
    tests = significance_tests(m)
    gap = generalisation_gap(m)
    models = models_present(m)
    sim, sim_auc = similarity_analysis(df, models)

    fig_dataset(df)
    fig_scaffolds(df, sim)
    fig_heatmap(roc_mean, roc_std, "ROC-AUC", "Endpoint-wise test ROC-AUC, scaffold split (mean of 3 seeds; bold = best per endpoint)",
                "fig3_endpoint_roc_auc.png")
    fig_heatmap(pr_mean, pr_std, "PR-AUC", "Endpoint-wise test PR-AUC, scaffold split (first row = prevalence = PR-AUC of a random classifier)",
                "fig4_endpoint_pr_auc.png", prevalence=prevalence)
    fig_summary(s, fam)
    fig_gap(gap, fam)
    show = [x for x in ["RandomForest", "XGBoost", "FP-MLP", "GCN", "AttentiveFP", "Ensemble RF+GNN"] if x in models]
    fig_similarity(sim_auc, fam, show)
    fig_learning_curves()

    print(s.to_string(index=False))
    print("\nMedian nearest-train similarity:", sim.groupby("split")["sim"].median().to_dict())
    print(json.dumps(json.loads((DATA_PROCESSED / "cleaning_stats.json").read_text())))


if __name__ == "__main__":
    main()
