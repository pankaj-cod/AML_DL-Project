# Tox21 toxicity prediction: one-command local setup and runs.
# Needs uv (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh

PY      := .venv/bin/python
DATA    := data/raw/tox21.csv.gz
URL     := https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz

.DEFAULT_GOAL := help
.PHONY: help setup data quick ml dl results notebook all view clean

help:  ## Show the available commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

setup: .venv/.installed $(DATA)  ## Create the Python 3.12 env and download Tox21 (first time only)

.venv/.installed: requirements.txt
	@command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv venv .venv --python 3.12 --allow-existing
	uv pip install --python $(PY) -r requirements.txt
	$(PY) -m ipykernel install --sys-prefix --name tox21 --display-name "Python (tox21)" >/dev/null
	@touch $@

$(DATA):
	@mkdir -p data/raw
	curl -sSL -o $@ $(URL)

data: setup  ## Clean, deduplicate and split the data (~15 s)
	$(PY) -m src.data

quick: data  ## Fast run from saved predictions: tables + figures (~1 min)
	$(PY) -m src.ensemble
	$(PY) -m src.analysis

ml: data  ## Phase 1: train Logistic Regression, Random Forest, XGBoost (~25 min)
	$(PY) -m src.baselines

dl: data  ## Phase 2: train the graph neural networks, 3 seeds in parallel (~1.5 h)
	$(PY) -W ignore -m src.gnn --seeds 0 --threads 2 --tag _seed0 & \
	$(PY) -W ignore -m src.gnn --seeds 1 --threads 2 --tag _seed1 & \
	$(PY) -W ignore -m src.gnn --seeds 2 --threads 2 --tag _seed2 & \
	wait

results: quick  ## Same as quick

notebook: setup  ## Re-execute the report notebook
	.venv/bin/jupyter-nbconvert --to notebook --execute --inplace notebooks/Tox21_Toxicity_Report.ipynb

all: ml dl quick notebook  ## Full pipeline from scratch (~2 h)

view:  ## Open the HTML report in the browser
	open report/Tox21_Toxicity_Report.html 2>/dev/null || xdg-open report/Tox21_Toxicity_Report.html

clean:  ## Remove caches (keeps data, results and the environment)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints
