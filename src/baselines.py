"""AML baselines: one model per endpoint on engineered chemical features.

Models: Logistic Regression, Random Forest, XGBoost (+ an XGBoost fingerprint-only
ablation). For every endpoint:
  * only molecules with an observed label for that endpoint are used (missing
    labels are excluded, never imputed);
  * class imbalance is handled endpoint by endpoint (class_weight='balanced' or
    scale_pos_weight = #neg / #pos computed on that endpoint's training labels);
  * hyper-parameters are chosen on the validation partition (ROC-AUC) and the
    selected model is scored once on the test partition.

Run:  python -m src.baselines
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .config import RESULTS, SPLIT_SEEDS, SPLIT_TYPES, TASKS
from .data import get_split, label_matrix, load_tox21
from .features import DescriptorScaler, load_engineered
from .metrics import endpoint_metrics

warnings.filterwarnings("ignore")
PRED_DIR = RESULTS / "preds"
PRED_DIR.mkdir(exist_ok=True)

GRIDS = {
    "LogReg": {"C": [0.01, 0.1, 1.0]},
    "RandomForest": {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.1]},
    "XGBoost": {"max_depth": [4, 6], "colsample_bytree": [0.3, 0.6]},
    "XGBoost-FPonly": {"max_depth": [4, 6], "colsample_bytree": [0.3, 0.6]},
}


def make_model(name, params, n_pos, n_neg, seed):
    if name == "LogReg":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000,
                                  solver="lbfgs")
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed, **params)
    return XGBClassifier(n_estimators=1000, learning_rate=0.05, subsample=0.8,
                         min_child_weight=1, reg_lambda=1.0, tree_method="hist",
                         scale_pos_weight=n_neg / max(n_pos, 1), eval_metric="auc",
                         early_stopping_rounds=50, n_jobs=8, random_state=seed, **params)


def fit_endpoint(name, Xtr, ytr, Xva, yva, seed):
    """Grid search on validation ROC-AUC; return the best fitted model."""
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    best = (-np.inf, None, None)
    keys, values = zip(*GRIDS[name].items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = make_model(name, params, n_pos, n_neg, seed)
        if name.startswith("XGBoost"):
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        auc = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
        if auc > best[0]:
            best = (auc, params, model)
    return best


def run_split(df, fp, desc, split_type, seed):
    y, mask = label_matrix(df)
    tr, va, te = get_split(df, split_type, seed)
    scaler = DescriptorScaler().fit(desc[tr])  # fitted on train only
    X_full = np.hstack([fp.astype(np.float32), scaler.transform(desc)])
    X_fp = fp.astype(np.float32)

    rows, chosen = [], []
    for name in GRIDS:
        if name == "XGBoost-FPonly" and split_type != "scaffold":
            continue
        X = X_fp if name == "XGBoost-FPonly" else X_full
        t0 = time.time()
        preds_va = np.full((len(va), len(TASKS)), np.nan, dtype=np.float32)
        preds_te = np.full((len(te), len(TASKS)), np.nan, dtype=np.float32)
        for t, task in enumerate(TASKS):
            mtr, mva = mask[tr, t], mask[va, t]
            itr, iva = tr[mtr], va[mva]
            val_auc, params, model = fit_endpoint(name, X[itr], y[itr, t], X[iva], y[iva, t], seed)
            preds_va[:, t] = model.predict_proba(X[va])[:, 1]
            preds_te[:, t] = model.predict_proba(X[te])[:, 1]
            chosen.append({"model": name, "split": split_type, "seed": seed,
                           "endpoint": task, "val_roc_auc": val_auc, **{k: str(v) for k, v in params.items()}})
        np.savez(PRED_DIR / f"{split_type}_{seed}_{name}.npz", test_idx=te, valid_idx=va,
                 test=preds_te, valid=preds_va)
        for r in endpoint_metrics(y[te], preds_te):
            rows.append({"model": name, "family": "Fingerprint AML", "split": split_type,
                         "seed": seed, **r})
        mean_auc = np.nanmean([r["roc_auc"] for r in rows if r["model"] == name and r["seed"] == seed])
        print(f"  {split_type}/{seed} {name:15s} test mean ROC-AUC={mean_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return rows, chosen


def main():
    df = load_tox21()
    fp, desc, _ = load_engineered(df)
    all_rows, all_chosen = [], []
    for split_type in SPLIT_TYPES:
        for seed in SPLIT_SEEDS:
            rows, chosen = run_split(df, fp, desc, split_type, seed)
            all_rows += rows
            all_chosen += chosen
            pd.DataFrame(all_rows).to_csv(RESULTS / "baseline_metrics.csv", index=False)
            pd.DataFrame(all_chosen).to_csv(RESULTS / "baseline_hparams.csv", index=False)


if __name__ == "__main__":
    main()
