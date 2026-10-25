#!/usr/bin/env bash
# Reproduce every result, table and figure from scratch (~1.5-2 h on an 8-core laptop CPU).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
fi
PY=.venv/bin/python

mkdir -p data/raw
[ -f data/raw/tox21.csv.gz ] || curl -sSL -o data/raw/tox21.csv.gz \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz

$PY -m ipykernel install --sys-prefix --name tox21 --display-name "Python (tox21)" >/dev/null
$PY -m src.data                       # clean, canonicalise, deduplicate, build splits
$PY -m src.baselines                  # LogReg / RF / XGBoost per endpoint
for s in 0 1 2; do                    # FP-MLP / GCN / GINE-MPNN / AttentiveFP / Hybrid
  $PY -W ignore -m src.gnn --seeds $s --threads 2 --tag _seed$s &
done
wait
$PY -m src.ensemble                   # late-fusion RF + best-validation GNN
$PY -m src.analysis                   # tables, statistics, figures
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace notebooks/Tox21_Toxicity_Report.ipynb
