"""Generates notebooks/Tox21_Toxicity_Report.ipynb (run, then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()

DISCUSSION = r"""
## 6. Discussion

### 6.1 Do learned graph representations beat engineered fingerprints?
**Not on their own, at this data scale.** On scaffold splits the best single model is the
fingerprint + descriptor **Random Forest** (ROC-AUC 0.834 ± 0.033, PR-AUC 0.496 ± 0.054). It is
significantly better than *every* graph network (paired Wilcoxon over 36 endpoint × seed pairs: vs GCN
+0.017, p = 0.004; vs AttentiveFP +0.017, p = 0.0001; vs GINE-MPNN +0.031, p = 0.0001). The best graph
networks (GCN 0.817, AttentiveFP 0.818) are statistically **indistinguishable from XGBoost (0.816) and
from the FP-MLP (0.814)** (all p > 0.3). The gap the brief hypothesised, that fixed substructure
dictionaries miss task-specific interactions, is real but small. About 6k training molecules are not enough
for message passing to learn a representation clearly better than ECFP4 plus 217 curated physicochemical
descriptors. This agrees with MoleculeNet (Wu et al., 2018) and Chemprop (Yang et al., 2019), where graph
models only pull clearly ahead of fingerprints on larger datasets or with pre-training.

### 6.2 But the two representations are complementary
* Graph models win individual endpoints that fingerprints handle poorly. **NR-Aromatase** is best with GCN
  (0.804 vs 0.750 for RF). AttentiveFP is the best single model on **NR-PPAR-gamma** (0.875 vs 0.870).
* The **late-fusion ensemble RF+GNN is the best model overall**: ROC-AUC **0.844 ± 0.025**, PR-AUC
  **0.503 ± 0.033** on scaffold splits. It beats RF alone on 26 of 36 endpoint × seed pairs (p = 0.001) and
  is best on 8 of 12 endpoints for ROC-AUC and on 7 of 12 for PR-AUC. The mixing weights, chosen on
  validation only, put on average **0.46 of the weight on the graph model**. If the GNN carried no
  information beyond the fingerprint, these weights would collapse to 0.
* **End-to-end fusion is harder than late fusion.** Even after the modality-dropout fix, the joint Hybrid
  network (0.809) does not beat its components. The two branches learn at very different speeds (see the
  learning curves: the FP-MLP's training loss collapses within 10 epochs while the graph models need 40–90),
  so a single early-stopping point is a compromise for both.

### 6.3 Scaffold generalisation
* **The scaffold split is the honest test.** Every model scores lower on scaffold splits than on random
  splits (Fig. 5–6). Under random splitting the median test molecule has a Tanimoto similarity of 0.58 to
  its nearest training molecule, against 0.45 under scaffold splitting. **7.1%** of random-split test
  molecules have a fingerprint *identical* to a training molecule (mostly stereoisomers, which are different
  canonical SMILES but have the same non-chiral ECFP), against 0.4% for scaffold splits.
* **Graph networks lose more when the chemistry is new.** The random → scaffold drop is 0.028–0.030 ROC-AUC
  for GCN, GINE-MPNN and AttentiveFP, but only 0.014 for RF and 0.016 for XGBoost. On *random* splits
  GCN (0.847) and AttentiveFP (0.847) tie RF (0.849). So part of the GNNs' apparent strength on random
  splits comes from interpolating between close analogues, which a scaffold split removes.
* **Accuracy versus distance to the training set (Fig. 7).** All models are good on molecules with a close
  training neighbour (Tanimoto ≥ 0.7: ROC-AUC 0.85–0.88) and degrade sharply for genuinely novel chemistry
  (< 0.3: 0.64–0.72). In that hardest bin RF (0.718) and GINE-MPNN (0.706) hold up best among single models,
  XGBoost degrades most (0.637), and the ensemble is best (0.723). This is the practically relevant regime
  for screening new chemical series, and no model is reliable there.
* **Endpoint level.** The stress-response assays (SR-ARE, SR-ATAD5, SR-MMP, SR-p53) and NR-AhR /
  NR-Aromatase have the largest scaffold gaps (≈ 0.055–0.068 averaged over models). They are driven by
  scaffold-specific chemotypes, so they are harder to extrapolate. NR-ER and NR-AR show *negative* gaps,
  which reflects the high variance of their small positive sets across splits rather than easier
  extrapolation.

### 6.4 Other observations
* **Descriptors matter as much as the model.** XGBoost on fingerprints only reaches 0.751, against 0.816 with
  descriptors added. Global properties such as logP, molecular weight and charge are strong toxicity
  signals that a GNN must learn from scratch from atom-level features.
* **More expressive is not better here.** The bond-aware GINE-MPNN is significantly *worse* than the plain GCN
  (−0.014, p = 0.007) and has the lowest PR-AUC of the graph models. Its extra capacity overfits the small
  training set.
* **Split variance is larger than most model differences.** The standard deviation across the three scaffold
  seeds (0.02–0.04) exceeds most pairwise model gaps. A single-split comparison could easily rank the models
  differently, which is why we report mean ± std and use paired tests.
* **PR-AUC tells a harsher story than ROC-AUC.** For rare endpoints (SR-ATAD5 at 4% prevalence,
  NR-PPAR-gamma at 3%) PR-AUC stays at 0.2–0.37 even when ROC-AUC exceeds 0.84. Ranking actives *first*
  is still hard, which matters for prioritising compounds for wet-lab testing.

### 6.5 Limitations and future work
* The baselines got a per-endpoint grid search, while the networks used one fixed architecture and setting
  each. A larger GNN search might narrow the gap.
* No self-supervised pre-training on large unlabelled molecule sets (e.g. GROVER, pretrained GIN). This is the
  most promising route to make graph representations beat fingerprints at this data size.
* Three seeds per split type; uncertainty estimates would tighten with more.
* Class-weighted losses optimise ranking, not calibration. Probabilities would need recalibration (e.g.
  Platt / isotonic on the validation set) before being used as risk scores.
* Eight SMILES that newer RDKit versions reject were dropped. Stereoisomers were kept as separate compounds.

## 7. Conclusion
On Tox21 under a strict scaffold split, **engineered fingerprints plus descriptors with a Random Forest
remain the strongest single model**. Message passing networks are competitive with XGBoost and a deep
fingerprint MLP but do not surpass them, and they degrade more when moving to unseen scaffolds. However,
the graph models capture **complementary information**: a validation-weighted late fusion of RF and a GNN
is the best model overall (scaffold ROC-AUC 0.844, PR-AUC 0.503) and wins most endpoints. In this
low-data regime the practical recommendation is to **combine** engineered and learned representations, not
to replace one with the other.
"""

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Molecular Toxicity Prediction via Fingerprints and Molecular Graph Networks
### AML / DL Group Project 8: Tox21, 12 toxicity endpoints

**Question.** Do *learned* molecular-graph representations (message passing neural networks)
predict toxicity-assay activity better than *engineered* chemical fingerprints, especially on
**chemical families never seen in training**?

**Approach (the "hybrid" logic).**
1. **AML baseline:** SMILES → Morgan fingerprints (ECFP4, 2048 bits) + 200+ RDKit descriptors → one
   Logistic Regression / Random Forest / XGBoost model **per endpoint**.
2. **Gap:** fingerprints encode a *fixed* dictionary of circular substructures. They cannot adapt to
   the endpoint, and they do not share information across the 12 related assays.
3. **DL model:** graph neural networks where atoms are nodes and bonds are edges (GCN, GINE message
   passing with bond features, AttentiveFP), trained **multitask with a masked loss** so missing labels
   contribute nothing. We add two controls: an FP-MLP (same fingerprints, deep multitask) and a
   **Hybrid** (graph embedding + fingerprint).
4. **Fair comparison:** identical cleaned data, identical **scaffold splits** (3 seeds), train-only
   preprocessing, validation-only model selection, and endpoint-wise **ROC-AUC and PR-AUC** that
   exclude missing labels. The same models are also run on **random splits** to measure the
   *scaffold generalisation gap*.

Everything below is reproducible with `./run_all.sh`. This notebook reads the saved results and
regenerates every table and figure.
""")

code("""
import sys, json, inspect, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from src.config import TASKS, RESULTS, FIGURES, DATA_PROCESSED
from src.data import load_tox21, label_matrix, get_split
from src import analysis, gnn

pd.set_option("display.precision", 3)
pd.set_option("display.width", 200)
""")

md("""
## 1. Data preparation

**Source:** Tox21 from MoleculeNet / DeepChem (`tox21.csv.gz`): 7,831 rows, 12 binary endpoints
(7 nuclear-receptor `NR-*` assays and 5 stress-response `SR-*` assays).

| Step | What we do | Why |
|---|---|---|
| Parse | RDKit `MolFromSmiles`, drop unparseable rows | invalid structures cannot be featurised |
| Standardise | RDKit cleanup → keep the **largest organic fragment** → canonical SMILES | strips salts and counter-ions so the same parent compound has one identity |
| **Deduplicate before splitting** | group by canonical SMILES. Agreeing labels are merged, **conflicting labels become missing** | prevents the same molecule appearing in both train and test (leakage) |
| Missing labels | kept as `NaN` → boolean **mask** | ~17% of labels are missing. We never impute them as negatives |
| Scaffolds | Bemis–Murcko scaffold per molecule | used for scaffold splitting |
""")

code("""
df = load_tox21()
stats = json.loads((DATA_PROCESSED / "cleaning_stats.json").read_text())
display(pd.Series({k: v for k, v in stats.items() if not isinstance(v, dict)}, name="value").to_frame())
print("Conflicting duplicate labels set to missing, per endpoint:")
print(stats["conflicting_labels_set_missing"])
df.head()
""")

code("""
y, mask = label_matrix(df)
ep = pd.DataFrame({"labelled": mask.sum(0), "positives": np.nansum(y, 0).astype(int),
                   "missing_%": 100 * (1 - mask.mean(0))}, index=TASKS)
ep["positive_%"] = 100 * ep["positives"] / ep["labelled"]
ep["neg:pos (class weight)"] = (ep["labelled"] - ep["positives"]) / ep["positives"]
display(ep.round(2))
display(Image(FIGURES / "fig1_dataset.png"))
""")

md("""
The endpoints are **heavily imbalanced** (2–16% actives). This is why we (i) weight the positive class
endpoint by endpoint (`class_weight='balanced'` / `scale_pos_weight` for the baselines, `pos_weight`
in the masked BCE for the networks) and (ii) report **PR-AUC** next to ROC-AUC. PR-AUC is
sensitive to performance on the rare positive class, and its chance level equals the prevalence.
""")

md("""
## 2. Scaffold splitting: testing on unseen chemical families

A random split puts close analogues of training molecules into the test set, which inflates scores.
We use a **randomised Bemis–Murcko scaffold split** (80/10/10). Every scaffold group is assigned to
exactly one partition, so test molecules come from ring systems the model has never seen. Large groups
(acyclic molecules, benzene) are forced into train so they do not dominate the small test sets.
Three seeds give three different, equally strict splits. We report mean ± std over them.
""")

code("""
rows = []
for st in ["scaffold", "random"]:
    for seed in [0, 1, 2]:
        tr, va, te = get_split(df, st, seed)
        sc = df["scaffold"]
        rows.append({"split": st, "seed": seed, "train": len(tr), "valid": len(va), "test": len(te),
                     "test scaffolds": sc.iloc[te].nunique(),
                     "test scaffolds also in train": len(set(sc.iloc[tr]) & set(sc.iloc[te]))})
display(pd.DataFrame(rows))
display(Image(FIGURES / "fig2_scaffold_split.png"))
""")

code("""
from rdkit import Chem
from rdkit.Chem import Draw
top = df.loc[df["scaffold"] != "", "scaffold"].value_counts().head(8)
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top.index], molsPerRow=8, subImgSize=(150, 120),
                     legends=[f"n={n}" for n in top.values])
""")

md("""
## 3. Models

### 3.1 AML baselines (engineered features, one model per endpoint)
* **Features:** Morgan/ECFP4 fingerprint (radius 2, 2048 bits) plus all RDKit 2D descriptors. Descriptors
  get a signed-log transform, median imputation and standardisation **fitted on the training split only**.
* **Models:** Logistic Regression (L2, C grid), Random Forest (500 trees, leaf/feature grid), XGBoost
  (depth/column-sampling grid, early stopping on validation).
* For each endpoint only molecules **with an observed label** are used. Hyper-parameters are chosen by
  validation ROC-AUC, and the test set is touched once.
* **Ablation:** XGBoost on fingerprints only (`XGBoost-FPonly`) isolates the value of the descriptors.

### 3.2 DL models (multitask, masked loss)
* **Graph:** 56-d atom features (element, degree, charge, hybridisation, #H, chirality, aromaticity,
  ring, mass) and 12-d bond features (type, stereo, conjugation, ring). Edges are bidirectional.
* **GCN:** 4 graph-convolution layers with residuals and batch-norm, mean+max readout.
* **GINE-MPNN:** 4 GIN layers whose messages are conditioned on **bond features**, sum+mean readout.
* **AttentiveFP:** attention-based message passing plus a GRU graph readout (Xiong et al., 2019).
* **FP-MLP (control):** same engineered features as the AML models, but a deep multitask network.
* **Hybrid (end-to-end):** GINE-MPNN graph embedding concatenated with the fingerprint/descriptor vector.
  A naive version early-stopped after 3–30 epochs because the fingerprint branch memorises the training set
  long before the graph branch learns (val ROC-AUC 0.78, *below* either component). The final version uses
  input dropout, a narrow projection, stronger L2 on the fingerprint branch, and **modality dropout** (the
  fingerprint embedding is zeroed for a random 50% of training molecules), which moved the best epoch to
  ~60 and raised val ROC-AUC on scaffold seed 0 from 0.784 to 0.829.
* **Ensemble RF+GNN (late fusion):** per endpoint, a rank-average of Random Forest and the graph model with the
  best *validation* ROC-AUC on that split. The mixing weight w ∈ {0, .25, .5, .75, 1} is chosen on validation.
* All networks: 12 sigmoid outputs, AdamW, early stopping on mean validation ROC-AUC, LR decay on plateau.

**Masked loss.** Missing labels are excluded from the loss. Positive-class weights are computed per
endpoint from the training labels:
""")

code("""print(inspect.getsource(gnn.masked_bce))""")

md("""
## 4. Results: endpoint-wise comparison on scaffold splits

Metrics are computed **per endpoint on observed labels only**, then averaged. Values are the mean over 3
scaffold-split seeds.
""")

code("""
m = analysis.load_metrics()
s = analysis.summary_table(m)
fmt = lambda a, b: a.map("{:.3f}".format) + " ± " + b.map("{:.3f}".format)
tab = s.assign(ROC_AUC=fmt(s.roc_auc_mean, s.roc_auc_std), PR_AUC=fmt(s.pr_auc_mean, s.pr_auc_std))
summary = tab.pivot(index=["model", "family"], columns="split", values=["ROC_AUC", "PR_AUC"])
summary = summary.reindex(analysis.MODEL_ORDER, level=0)
summary
""")

code("""
roc, roc_sd = analysis.endpoint_table(m, "roc_auc")
display(Markdown("**Test ROC-AUC per endpoint (scaffold split, mean of 3 seeds)**"))
display(roc.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold"))
display(Image(FIGURES / "fig3_endpoint_roc_auc.png"))
""")

code("""
pr, pr_sd = analysis.endpoint_table(m, "pr_auc")
display(Markdown("**Test PR-AUC per endpoint (scaffold split, mean of 3 seeds)**"))
display(pr.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold"))
display(Image(FIGURES / "fig4_endpoint_pr_auc.png"))
""")

code("""
best_by_endpoint = pd.DataFrame({"best ROC-AUC model": roc.idxmax(), "ROC-AUC": roc.max(),
                                 "best PR-AUC model": pr.idxmax(), "PR-AUC": pr.max()})
display(best_by_endpoint)
fam = analysis.family_of(m)
print("Endpoints won (ROC-AUC) by family:", best_by_endpoint["best ROC-AUC model"].map(fam).value_counts().to_dict())
print("Endpoints won (PR-AUC)  by family:", best_by_endpoint["best PR-AUC model"].map(fam).value_counts().to_dict())
""")

md("""
### 4.1 Is the difference significant?
Paired **Wilcoxon signed-rank tests** over the 36 (endpoint × seed) pairs of test ROC-AUC. A positive
`mean_diff` means model A is better.
""")

code("""
tests = analysis.significance_tests(m)
key = ["RandomForest", "XGBoost", "FP-MLP", "GCN", "GINE-MPNN", "AttentiveFP", "Hybrid"]
t = tests[tests.model_a.isin(key) & tests.model_b.isin(key)].sort_values("p_value")
t.style.format({"mean_diff_roc_auc": "{:+.4f}", "p_value": "{:.4f}"})
""")

md("""
## 5. Scaffold generalisation analysis

Three complementary views:
1. **Random vs scaffold split:** how much each model's ROC-AUC drops when the test chemistry is new.
2. **Per-endpoint gap:** which assays are hardest to extrapolate.
3. **Accuracy vs distance to the training set:** each test molecule is binned by the Tanimoto similarity
   to its nearest training neighbour (test sets of all 6 splits pooled).
""")

code("""display(Image(FIGURES / "fig5_summary_scaffold_vs_random.png"))""")

code("""
gap = analysis.generalisation_gap(m)
display(Markdown("**ROC-AUC(random) − ROC-AUC(scaffold), per endpoint**"))
display(gap.assign(mean=gap.mean(axis=1)).style.format("{:+.3f}").background_gradient(cmap="Oranges", axis=None))
display(Image(FIGURES / "fig6_generalisation_gap.png"))
""")

code("""
sim_auc = pd.read_csv(RESULTS / "auc_by_similarity.csv")
display(sim_auc.pivot(index="model", columns="bin", values="mean_roc_auc")
        .reindex([x for x in analysis.MODEL_ORDER if x in set(sim_auc.model)])
        [["<0.3", "0.3–0.4", "0.4–0.5", "0.5–0.7", "≥0.7"]].style.format("{:.3f}"))
display(Image(FIGURES / "fig7_auc_vs_similarity.png"))
""")

code("""display(Image(FIGURES / "fig8_learning_curves.png"))""")

md(DISCUSSION)

md("""
## 7. References
1. Wu, Z. et al. (2018). *MoleculeNet: a benchmark for molecular machine learning.* Chemical Science, 9, 513–530.
2. Rogers, D. & Hahn, M. (2010). *Extended-connectivity fingerprints.* J. Chem. Inf. Model., 50, 742–754.
3. Bemis, G. W. & Murcko, M. A. (1996). *The properties of known drugs. 1. Molecular frameworks.* J. Med. Chem., 39, 2887–2893.
4. Kipf, T. N. & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks.* ICLR.
5. Gilmer, J. et al. (2017). *Neural message passing for quantum chemistry.* ICML.
6. Xu, K. et al. (2019). *How powerful are graph neural networks?* ICLR. Hu, W. et al. (2020). *Strategies for pre-training graph neural networks* (GINE). ICLR.
7. Xiong, Z. et al. (2019). *Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism (AttentiveFP).* J. Med. Chem., 63, 8749–8760.
8. Yang, K. et al. (2019). *Analyzing learned molecular representations for property prediction (Chemprop, D-MPNN).* J. Chem. Inf. Model., 59, 3370–3388.
9. Chen, T. & Guestrin, C. (2016). *XGBoost: a scalable tree boosting system.* KDD.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "tox21", "display_name": "Python (tox21)", "language": "python"}
out = Path(__file__).parent / "Tox21_Toxicity_Report.ipynb"
nbf.write(nb, out)
print("wrote", out)
