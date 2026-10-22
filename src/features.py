"""Molecular featurisation.

* Engineered features for the AML baselines: Morgan (ECFP4) bit fingerprints plus
  the full set of RDKit 2D physicochemical descriptors.
* Graph features for the DL models: atoms are nodes, bonds are (bidirectional)
  edges, each with a one-hot / binary feature vector.
"""
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from .config import DATA_PROCESSED, MORGAN_BITS, MORGAN_RADIUS

# --------------------------------------------------------------------------- #
# Engineered features
# --------------------------------------------------------------------------- #
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def morgan_fp(smiles_list):
    """(n, MORGAN_BITS) uint8 matrix and the RDKit bit vectors (for Tanimoto)."""
    bitvecs = [_morgan.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
    X = np.zeros((len(bitvecs), MORGAN_BITS), dtype=np.uint8)
    for i, bv in enumerate(bitvecs):
        DataStructs.ConvertToNumpyArray(bv, X[i])
    return X, bitvecs


def rdkit_descriptors(smiles_list):
    names = [n for n, _ in Descriptors.descList]
    X = np.zeros((len(smiles_list), len(names)), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        d = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
        X[i] = [d[n] for n in names]
    return X, names


def load_engineered(df):
    """Cached Morgan fingerprints and raw descriptors."""
    path = DATA_PROCESSED / "engineered_features.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["fp"], z["desc"], list(z["desc_names"])
    fp, _ = morgan_fp(df["smiles"])
    desc, names = rdkit_descriptors(df["smiles"])
    np.savez_compressed(path, fp=fp, desc=desc, desc_names=np.array(names))
    return fp, desc, names


class DescriptorScaler:
    """Train-only preprocessing of descriptors: signed log for heavy tails,
    median imputation, standardisation and clipping. Fitted on the training
    partition only to avoid leakage into validation / test."""

    def fit(self, X):
        X = self._log(X)
        self.median_ = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.median_, X)
        self.keep_ = X.std(axis=0) > 1e-8  # drop constant columns
        self.scaler_ = StandardScaler().fit(X[:, self.keep_])
        return self

    def transform(self, X):
        X = self._log(X)
        X = np.where(np.isnan(X), self.median_, X)
        return np.clip(self.scaler_.transform(X[:, self.keep_]), -10, 10).astype(np.float32)

    @staticmethod
    def _log(X):
        X = np.where(np.isfinite(X), X, np.nan)
        return np.sign(X) * np.log1p(np.abs(X))


# --------------------------------------------------------------------------- #
# Graph features
# --------------------------------------------------------------------------- #
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se",
              "Na", "K", "Ca", "Fe", "Zn", "Cu", "Sn", "Hg", "As", "Al"]
HYBRID = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
          Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
          Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
CHIRAL = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
          Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
STEREO = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
          Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOANY]


def _one_hot(x, choices):
    """One-hot with a trailing 'other' slot."""
    v = [0.0] * (len(choices) + 1)
    v[choices.index(x) if x in choices else -1] = 1.0
    return v


def atom_features(a):
    return (_one_hot(a.GetSymbol(), ATOM_TYPES)
            + _one_hot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _one_hot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + _one_hot(a.GetHybridization(), HYBRID)
            + _one_hot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + _one_hot(a.GetChiralTag(), CHIRAL)
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0])


def bond_features(b):
    return (_one_hot(b.GetBondType(), BOND_TYPES)
            + _one_hot(b.GetStereo(), STEREO)
            + [float(b.GetIsConjugated()), float(b.IsInRing())])


N_ATOM_FEATURES = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
N_BOND_FEATURES = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, ef = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]
        dst += [j, i]
        ef += [f, f]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(ef, dtype=torch.float)
    else:  # single-atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, N_BOND_FEATURES), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float).view(1, -1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_graphs(df, y):
    path = DATA_PROCESSED / "graphs.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    graphs = [mol_to_graph(s, y[i]) for i, s in enumerate(df["smiles"])]
    torch.save(graphs, path)
    return graphs
