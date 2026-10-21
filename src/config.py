"""Shared paths, endpoint names and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "tox21.csv.gz"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Every model is evaluated on the same splits.
# "scaffold" = randomised (balanced) Bemis-Murcko scaffold split, "random" = molecule-level random split.
SPLIT_SEEDS = [0, 1, 2]
SPLIT_TYPES = ["scaffold", "random"]
FRAC_TRAIN, FRAC_VALID, FRAC_TEST = 0.8, 0.1, 0.1

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

for p in (DATA_PROCESSED, RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
