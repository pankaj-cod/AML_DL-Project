"""Endpoint-wise metrics that ignore missing labels."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import TASKS


def endpoint_metrics(y_true, y_score):
    """Per-endpoint ROC-AUC and PR-AUC computed only on observed labels.

    y_true contains NaN for missing labels. An endpoint whose observed labels
    contain a single class is reported as NaN (metric undefined).
    """
    rows = []
    for t, name in enumerate(TASKS):
        m = ~np.isnan(y_true[:, t])
        yt, ys = y_true[m, t], y_score[m, t]
        n_pos = int(yt.sum())
        ok = 0 < n_pos < len(yt)
        rows.append({
            "endpoint": name,
            "n": int(m.sum()),
            "n_pos": n_pos,
            "prevalence": n_pos / max(len(yt), 1),
            "roc_auc": roc_auc_score(yt, ys) if ok else np.nan,
            "pr_auc": average_precision_score(yt, ys) if ok else np.nan,
        })
    return rows


def mean_roc_auc(y_true, y_score):
    return float(np.nanmean([r["roc_auc"] for r in endpoint_metrics(y_true, y_score)]))
