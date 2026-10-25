"""Late-fusion hybrid: combine the best fingerprint model with the best graph model.

For every split, the fingerprint partner is Random Forest and the graph partner is
the graph network with the highest *validation* ROC-AUC on that split. Per
endpoint, the two models' scores are converted to ranks (so differently
calibrated probabilities are comparable) and mixed as

    score = w * rank(graph) + (1 - w) * rank(fingerprint),  w in {0, .25, .5, .75, 1},

with w chosen on the validation partition. The test set is only used for the
final score, so the comparison stays fair.

Run:  python -m src.ensemble   (after src.baselines and src.gnn)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .config import RESULTS, SPLIT_SEEDS, SPLIT_TYPES, TASKS
from .data import label_matrix, load_tox21
from .metrics import endpoint_metrics, mean_roc_auc

PRED_DIR = RESULTS / "preds"
FP_MODEL = "RandomForest"
GRAPH_MODELS = ["GCN", "GINE-MPNN", "AttentiveFP"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
NAME = "Ensemble RF+GNN"


def _rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def _auc(y, p):
    m = ~np.isnan(y)
    return roc_auc_score(y[m], p[m]) if 0 < np.nansum(y) < m.sum() else np.nan


def main():
    df = load_tox21()
    y, _ = label_matrix(df)
    rows, choices = [], []
    for st in SPLIT_TYPES:
        for seed in SPLIT_SEEDS:
            fp = np.load(PRED_DIR / f"{st}_{seed}_{FP_MODEL}.npz")
            va, te = fp["valid_idx"], fp["test_idx"]
            graph = {g: np.load(PRED_DIR / f"{st}_{seed}_{g}.npz") for g in GRAPH_MODELS}
            g_best = max(graph, key=lambda g: mean_roc_auc(y[va], graph[g]["valid"]))
            gp = graph[g_best]
            out_va = np.zeros_like(fp["valid"])
            out_te = np.zeros_like(fp["test"])
            for t, task in enumerate(TASKS):
                rv_fp, rv_g = _rank(fp["valid"][:, t]), _rank(gp["valid"][:, t])
                w = max(WEIGHTS, key=lambda w: _auc(y[va, t], w * rv_g + (1 - w) * rv_fp))
                out_va[:, t] = w * rv_g + (1 - w) * rv_fp
                out_te[:, t] = w * _rank(gp["test"][:, t]) + (1 - w) * _rank(fp["test"][:, t])
                choices.append({"split": st, "seed": seed, "endpoint": task,
                                "graph_partner": g_best, "graph_weight": w})
            np.savez(PRED_DIR / f"{st}_{seed}_{NAME}.npz", test_idx=te, valid_idx=va,
                     test=out_te, valid=out_va)
            for r in endpoint_metrics(y[te], out_te):
                rows.append({"model": NAME, "family": "Hybrid (graph + FP)", "split": st,
                             "seed": seed, **r})
            print(f"  {st}/{seed} partner={g_best:12s} test={mean_roc_auc(y[te], out_te):.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "ensemble_metrics.csv", index=False)
    pd.DataFrame(choices).to_csv(RESULTS / "ensemble_weights.csv", index=False)


if __name__ == "__main__":
    main()
