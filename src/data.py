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
