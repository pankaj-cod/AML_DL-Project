"""Loading, cleaning, deduplication and splitting of Tox21.

Pipeline
--------
1. Parse every SMILES with RDKit (unparseable rows are dropped and counted).
2. Standardise: RDKit cleanup -> keep the largest organic fragment (strips salts /
   counter-ions) -> canonical isomeric SMILES.
3. Deduplicate on the canonical SMILES *before* splitting, so the same compound can
   never sit in both train and test. Labels of duplicates are merged per endpoint:
   agreeing labels are kept, conflicting labels become missing (NaN).
4. Missing labels are preserved as NaN; a mask is derived from them downstream.
5. Bemis-Murcko scaffolds are computed for scaffold splitting.
"""
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import (DATA_PROCESSED, DATA_RAW, FRAC_TEST, FRAC_TRAIN, FRAC_VALID,
                     TASKS)

RDLogger.DisableLog("rdApp.*")

_fragment_chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def standardise_smiles(smiles):
    """Return the canonical SMILES of the parent (largest) fragment, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _fragment_chooser.choose(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _merge_labels(values):
    """Merge duplicate labels for one endpoint: agree -> keep, conflict -> NaN."""
    v = values.dropna()
    if len(v) == 0:
        return np.nan
    return float(v.iloc[0]) if v.nunique() == 1 else np.nan


def load_tox21(force=False):
    """Return the cleaned, deduplicated dataframe (cached to data/processed)."""
    out = DATA_PROCESSED / "tox21_clean.csv"
    stats_path = DATA_PROCESSED / "cleaning_stats.json"
    if out.exists() and not force:
        return pd.read_csv(out)

    raw = pd.read_csv(DATA_RAW)
    stats = {"raw_rows": len(raw)}
    raw["smiles_std"] = raw["smiles"].map(standardise_smiles)
    stats["unparseable"] = int(raw["smiles_std"].isna().sum())
    raw = raw.dropna(subset=["smiles_std"])

    n_labels_before = int(raw[TASKS].notna().sum().sum())
    grouped = raw.groupby("smiles_std", sort=False)
    stats["duplicate_rows_merged"] = int(len(raw) - grouped.ngroups)

    merged = grouped[TASKS].agg(_merge_labels)
    conflicts = {}
    for t in TASKS:
        g = grouped[t].agg(lambda v: v.dropna().nunique() > 1)
        conflicts[t] = int(g.sum())
    stats["conflicting_labels_set_missing"] = conflicts

    df = merged.reset_index().rename(columns={"smiles_std": "smiles"})
    df.insert(0, "mol_id", grouped["mol_id"].first().values)
    df["scaffold"] = df["smiles"].map(murcko_scaffold)
    # Drop molecules that ended up with no labels at all.
    df = df[df[TASKS].notna().any(axis=1)].reset_index(drop=True)

    stats["final_molecules"] = len(df)
    stats["labels_before_merge"] = n_labels_before
    stats["labels_after_merge"] = int(df[TASKS].notna().sum().sum())
    stats["unique_scaffolds"] = int(df["scaffold"].nunique())
    stats["acyclic_molecules"] = int((df["scaffold"] == "").sum())
    df.to_csv(out, index=False)
    stats_path.write_text(json.dumps(stats, indent=2))
    return df


def label_matrix(df):
    """Labels as float array with NaN for missing, plus the boolean observed mask."""
    y = df[TASKS].to_numpy(dtype=np.float32)
    return y, ~np.isnan(y)


def scaffold_split(df, seed, frac_train=FRAC_TRAIN, frac_valid=FRAC_VALID):
    """Randomised ("balanced") Bemis-Murcko scaffold split.

    Whole scaffold groups are assigned to a single partition, so test molecules
    always belong to chemical series never seen in training. Groups larger than
    half the test size are placed in train first (as in Chemprop's balanced
    scaffold split) so that the large acyclic / benzene groups do not swamp the
    small validation and test sets; the remaining groups are shuffled with
    `seed`, which gives a different but equally strict split per seed.
    """
    n = len(df)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    n_test = n - n_train - n_valid

    groups = defaultdict(list)
    for i, s in enumerate(df["scaffold"].fillna("")):
        groups[s].append(i)
    sets = list(groups.values())
    big = [s for s in sets if len(s) > n_test / 2]
    small = [s for s in sets if len(s) <= n_test / 2]
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train, valid, test = [], [], []
    for s in big + small:
        if len(train) + len(s) <= n_train:
            train += s
        elif len(valid) + len(s) <= n_valid:
            valid += s
        else:
            test += s
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def random_split(df, seed, frac_train=FRAC_TRAIN, frac_valid=FRAC_VALID):
    n = len(df)
    perm = np.random.RandomState(seed).permutation(n)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    return (np.sort(perm[:n_train]), np.sort(perm[n_train:n_train + n_valid]),
            np.sort(perm[n_train + n_valid:]))


def get_split(df, split_type, seed):
    """Cached split indices so every model sees identical partitions."""
    path = DATA_PROCESSED / f"split_{split_type}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return z["train"], z["valid"], z["test"]
    fn = scaffold_split if split_type == "scaffold" else random_split
    tr, va, te = fn(df, seed)
    np.savez(path, train=tr, valid=va, test=te)
    return tr, va, te


if __name__ == "__main__":
    from .config import SPLIT_SEEDS, SPLIT_TYPES
    df = load_tox21(force=True)
    print(json.dumps(json.loads((DATA_PROCESSED / "cleaning_stats.json").read_text()), indent=2))
    for st in SPLIT_TYPES:
        for seed in SPLIT_SEEDS:
            tr, va, te = get_split(df, st, seed)
            sc = df["scaffold"]
            overlap = len(set(sc.iloc[tr]) & set(sc.iloc[te]))
            print(f"{st:8s} seed={seed}: train={len(tr)} valid={len(va)} test={len(te)} "
                  f"train/test shared scaffolds={overlap}")
