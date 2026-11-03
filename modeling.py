# -*- coding: utf-8 -*-
"""Canonical feature engineering for the 60 gamma-glutamyl peptide dataset.

Feature blocks
--------------
desc      : RDKit 2D descriptors of the whole molecule
resid     : middle-residue side-chain descriptors + C-terminal group descriptors
term      : one-hot C-terminal class derived from structure
morgan_ap : hashed atom-pair fingerprint
cb        : ChemBERTa-77M-MLM mean-pooled SMILES embedding (384-d)
mf        : MoLFormer-XL mean-pooled SMILES embedding (768-d)
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "data", "raw", "60分子.xlsx")
DATA_XLSX = _RAW if os.path.exists(_RAW) else os.path.join(_ROOT, "60分子.xlsx")
CORE = Chem.MolFromSmarts("NC(CCC(=O)N)C(=O)O")
TERM_CLASSES = ["free", "Gly", "Glypeptide", "glycolic", "lactic", "bAla", "Ala", "otherAA"]

SIDE_DESC = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NumAliphaticRings", "NumSaturatedRings",
    "NumAliphaticHeterocycles", "NumAromaticHeterocycles",
]

_AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
_MC_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
_MC_CHIR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024,
                                                     includeChirality=True)
_DESC_LIST = list(Descriptors.descList)

_EMB_CACHE = {}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_dataset(path=DATA_XLSX):
    df = pd.read_excel(path, header=None)
    df.columns = ["name", "smiles", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["mol"] = df["smiles"].map(Chem.MolFromSmiles)
    if df["mol"].isna().any():
        bad = df.loc[df["mol"].isna(), "smiles"].tolist()
        raise ValueError(f"Invalid SMILES: {bad}")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def read_query(path):
    """Read a prediction file; accepts name/smiles or SMILES-only columns."""
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, header=None)
    else:
        raw = pd.read_csv(path, header=None)
    # drop a header row if present
    if len(raw) and any(str(x).strip().lower() in ("smiles", "name", "value")
                        for x in raw.iloc[0].tolist()):
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] >= 3:
        df = raw.iloc[:, :3].copy()
        df.columns = ["name", "smiles", "value"]
        df["name"] = df["name"].astype(str).str.strip()
    elif raw.shape[1] == 2:
        df = raw.copy()
        df.columns = ["name", "smiles"]
    else:
        df = raw.copy()
        df.columns = ["smiles"]
        df["name"] = range(1, len(df) + 1)
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["mol"] = df["smiles"].map(Chem.MolFromSmiles)
    bad = df["mol"].isna()
    if bad.any():
        raise ValueError(f"Invalid SMILES rows: {df.loc[bad, 'smiles'].tolist()}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# residue decomposition
# --------------------------------------------------------------------------
def _frag_from_atoms(mol, atom_set, marker):
    a = set(atom_set)
    cut = [b.GetIdx() for b in mol.GetBonds()
           if (b.GetBeginAtomIdx() in a) != (b.GetEndAtomIdx() in a)]
    if not cut:
        frag = Chem.Mol(mol)
    else:
        full = Chem.FragmentOnBonds(mol, cut, addDummies=True,
                                    dummyLabels=[(0, 0)] * len(cut))
        tuples = Chem.GetMolFrags(full)
        sel = next((t for t in tuples if marker in t), None)
        if sel is None:
            return None
        keep = set(sel)
        rw = Chem.RWMol(full)
        for idx in sorted(range(full.GetNumAtoms()), reverse=True):
            if idx not in keep:
                rw.RemoveAtom(idx)
        frag = rw.GetMol()
    rw = Chem.RWMol(frag)
    for idx in sorted([x.GetIdx() for x in rw.GetAtoms() if x.GetAtomicNum() == 0],
                      reverse=True):
        d = rw.GetAtomWithIdx(idx)
        nbr = d.GetNeighbors()[0]
        nbr.SetNoImplicit(False)
        rw.RemoveAtom(idx)
    f = rw.GetMol()
    Chem.SanitizeMol(f)
    return f


def _bfs_path(mol, start, targets):
    parent = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur in targets:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nbr.GetIdx()
            if j in parent or nbr.GetAtomicNum() == 1:
                continue
            parent[j] = cur
            dq.append(j)
    return None


def decompose_residue(mol):
    """Return (middle side-chain fragment, C-terminal fragment)."""
    core_idx = set(mol.GetSubstructMatch(CORE))
    match = mol.GetSubstructMatch(CORE)
    amide_n = match[6]
    cands = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if i in core_idx or atom.GetAtomicNum() != 6:
            continue
        if not any(b.GetBondTypeAsDouble() == 2
                   and b.GetOtherAtom(atom).GetAtomicNum() == 8
                   for b in atom.GetBonds()):
            continue
        hetero = [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds()
                  if b.GetOtherAtom(atom).GetAtomicNum() in (7, 8)
                  and b.GetBondTypeAsDouble() == 1]
        if hetero:
            cands.append(i)
    paths = [(len(p), c, p) for c in cands
             if (p := _bfs_path(mol, amide_n, {c})) is not None]
    if not paths:
        return None, None
    paths.sort(key=lambda x: x[0])
    _, carbonyl, path = paths[0]
    ca = path[1]
    path_set = set(path)

    term = None
    for b in mol.GetAtomWithIdx(carbonyl).GetBonds():
        j = b.GetOtherAtomIdx(carbonyl)
        if j in path_set:
            continue
        if (b.GetOtherAtom(mol.GetAtomWithIdx(carbonyl)).GetAtomicNum() in (7, 8)
                and b.GetBondTypeAsDouble() == 1):
            term = j
            break
    if term is None:
        return None, None

    banned = (path_set - {ca}) | core_idx
    side = {ca}
    stack = [ca]
    while stack:
        cur = stack.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nbr.GetIdx()
            if j in banned or j in side or nbr.GetAtomicNum() == 1:
                continue
            side.add(j)
            stack.append(j)

    termset = {term}
    stack = [term]
    while stack:
        cur = stack.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nbr.GetIdx()
            if j in termset or j == carbonyl or nbr.GetAtomicNum() == 1:
                continue
            termset.add(j)
            stack.append(j)

    return _frag_from_atoms(mol, side, ca), _frag_from_atoms(mol, termset, term)


def _frag_desc(f):
    out = {}
    if f is None or f.GetNumAtoms() == 0:
        return {k: 0.0 for k in SIDE_DESC} | {
            "nC": 0.0, "nN": 0.0, "nO": 0.0, "nS": 0.0, "nHal": 0.0,
            "nBranch": 0.0, "nAmide": 0.0, "nEster": 0.0, "nAcid": 0.0,
            "nMethyl": 0.0, "nStereoR": 0.0, "nStereoS": 0.0,
        }
    vals = {}
    for k in SIDE_DESC:
        try:
            v = float(getattr(Descriptors, k)(f))
            vals[k] = v if np.isfinite(v) else np.nan
        except Exception:
            vals[k] = np.nan
    vals["nC"] = sum(a.GetAtomicNum() == 6 for a in f.GetAtoms())
    vals["nN"] = sum(a.GetAtomicNum() == 7 for a in f.GetAtoms())
    vals["nO"] = sum(a.GetAtomicNum() == 8 for a in f.GetAtoms())
    vals["nS"] = sum(a.GetAtomicNum() == 16 for a in f.GetAtoms())
    vals["nHal"] = sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in f.GetAtoms())
    vals["nBranch"] = sum(
        1 for a in f.GetAtoms()
        if a.GetAtomicNum() == 6
        and sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) >= 3)
    vals["nAmide"] = len(f.GetSubstructMatches(
        Chem.MolFromSmarts("[NX3][CX3](=[OX1])")))
    vals["nEster"] = len(f.GetSubstructMatches(
        Chem.MolFromSmarts("[OX2;!H1][CX3](=[OX1])")))
    vals["nAcid"] = len(f.GetSubstructMatches(
        Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")))
    vals["nMethyl"] = sum(
        1 for a in f.GetAtoms()
        if a.GetAtomicNum() == 6
        and sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) == 1
        and all(n.GetAtomicNum() in (1, 6) for n in a.GetNeighbors()))
    try:
        centers = Chem.FindMolChiralCenters(f, includeUnassigned=False,
                                            useLegacyImplementation=False)
        vals["nStereoR"] = sum(1 for _, c in centers if c == "R")
        vals["nStereoS"] = sum(1 for _, c in centers if c == "S")
    except Exception:
        vals["nStereoR"] = 0.0
        vals["nStereoS"] = 0.0
    return {k: float(v) for k, v in vals.items()}


def residue_feature_row(mol):
    side, term = decompose_residue(mol)
    s = _frag_desc(side)
    t = _frag_desc(term)
    return {f"S_{k}": v for k, v in s.items()} | {f"T_{k}": v for k, v in t.items()}


def term_class(row):
    if row["T_nC"] == 0:
        return "free"
    if row["T_nN"] == 0:
        return "lactic" if row["T_nC"] >= 3 else "glycolic"
    if row["T_nN"] >= 2:
        return "Glypeptide"
    if row["T_nC"] == 2:
        return "Gly"
    if row["T_nC"] == 3:
        if row["T_nS"] > 0 or row["T_nO"] > 2:
            return "otherAA"
        return "Ala" if row["T_nMethyl"] > 0 else "bAla"
    return "otherAA"


# --------------------------------------------------------------------------
# feature selection
# --------------------------------------------------------------------------
class GainSelect(BaseEstimator, TransformerMixin):
    """Select top-k features by XGBoost gain importance (fit on training fold only)."""

    def __init__(self, k=300):
        self.k = k

    def fit(self, X, y):
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=2,
                             subsample=0.8, colsample_bytree=0.5,
                             reg_lambda=1.0, random_state=0, n_jobs=4,
                             verbosity=0)
        m.fit(X, y)
        score = m.get_booster().get_score(importance_type="gain")
        s = np.zeros(X.shape[1])
        for key, v in score.items():
            s[int(key[1:])] = v
        s = s + 1e-9 * np.asarray(X, dtype=np.float64).std(0)
        self.idx_ = np.argsort(-s)[: min(self.k, X.shape[1])]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


class ETImportanceSelect(BaseEstimator, TransformerMixin):
    """Select top-k classification features by ExtraTrees impurity importance."""

    def __init__(self, k=300, n_estimators=400):
        self.k = k
        self.n_estimators = n_estimators

    def fit(self, X, y):
        m = ExtraTreesClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=2,
            max_features=0.3, class_weight="balanced", n_jobs=4,
            random_state=0)
        m.fit(X, y)
        self.idx_ = np.argsort(-m.feature_importances_)[: min(self.k, X.shape[1])]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


# --------------------------------------------------------------------------
# 2D descriptors / fingerprints
# --------------------------------------------------------------------------
def whole_descriptors(mol):
    out = {}
    for name, fn in _DESC_LIST:
        try:
            v = float(fn(mol))
            out[name] = v if np.isfinite(v) else np.nan
        except Exception:
            out[name] = np.nan
    return out


def atom_pair(mol):
    return _AP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


# --------------------------------------------------------------------------
# pretrained embeddings
# --------------------------------------------------------------------------
def _load_hf(tag):
    if tag in _EMB_CACHE:
        return _EMB_CACHE[tag]
    import torch
    from transformers import AutoModel, AutoTokenizer
    paths = {
        "cb": "DeepChem/ChemBERTa-77M-MLM",
        "mf": "ibm-research/MoLFormer-XL-both-10pct",
    }
    trust = tag == "mf"
    kw = dict(trust_remote_code=trust, local_files_only=True)
    try:
        tok = AutoTokenizer.from_pretrained(paths[tag], **kw)
        model = AutoModel.from_pretrained(paths[tag], **kw)
    except Exception:
        tok = AutoTokenizer.from_pretrained(paths[tag], trust_remote_code=trust)
        model = AutoModel.from_pretrained(paths[tag], trust_remote_code=trust)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    _EMB_CACHE[tag] = (tok, model, dev)
    return _EMB_CACHE[tag]


def smiles_embeddings(smiles, tag="cb", batch_size=32, max_len=128):
    import torch
    tok, model, dev = _load_hf(tag)
    out = []
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy())
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------
# canonical feature matrix
# --------------------------------------------------------------------------
def build_feature_blocks(df, need_embeddings=("cb", "mf"), verbose=False):
    mols = df["mol"].tolist()
    if verbose:
        print("  residue features...", flush=True)
    resid = pd.DataFrame([residue_feature_row(m) for m in mols])
    cols = sorted(resid.columns)
    resid = resid[cols]
    classes = [term_class(r) for _, r in resid.iterrows()]
    termoh = pd.get_dummies(pd.Series(classes), dtype=float)
    termoh = termoh.reindex(columns=TERM_CLASSES, fill_value=0.0)
    if verbose:
        print("  descriptors...", flush=True)
    desc = pd.DataFrame([whole_descriptors(m) for m in mols])
    if verbose:
        print("  atom-pair fingerprints...", flush=True)
    ap = np.array([atom_pair(m) for m in mols], dtype=np.float32)
    morgan_count = np.array([_MC_GEN.GetCountFingerprintAsNumPy(m) for m in mols],
                            dtype=np.float32)
    morgan_chir = np.array([_MC_CHIR.GetFingerprintAsNumPy(m) for m in mols],
                           dtype=np.float32)
    maccs = np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols],
                     dtype=np.float32)
    blocks = {
        "morgan_count": morgan_count,
        "morgan_chir": morgan_chir,
        "maccs": maccs,
        "resid": resid.values.astype(np.float32),
        "resid_cols": np.array(cols, dtype=object),
        "term": termoh.values.astype(np.float32),
        "term_cols": np.array(TERM_CLASSES, dtype=object),
        "desc": desc.values.astype(np.float32),
        "desc_cols": np.array(desc.columns, dtype=object),
        "ap": ap,
        "classes": classes,
    }
    if "cb" in need_embeddings:
        if verbose:
            print("  ChemBERTa embedding...", flush=True)
        blocks["cb"] = smiles_embeddings(df["smiles"].tolist(), "cb")
    if "mf" in need_embeddings:
        if verbose:
            print("  MoLFormer embedding...", flush=True)
        blocks["mf"] = smiles_embeddings(df["smiles"].tolist(), "mf")
    return blocks


VIEWS = {
    "B": ["cb", "resid", "term", "ap"],
    "B_count": ["cb", "resid", "term", "ap", "morgan_count"],
    "B_maccs": ["cb", "resid", "term", "ap", "maccs"],
    "B_macchir": ["cb", "resid", "term", "ap", "maccs", "morgan_chir"],
    "B_fpall": ["cb", "resid", "term", "ap", "morgan_count", "maccs", "morgan_chir"],
    "residue_desc": ["resid", "term", "desc"],
    "A": ["cb", "resid", "term"],
    "D": ["cb", "mf", "resid", "term", "ap"],
}


def fusion_matrix(blocks, view="B"):
    """Assemble a named feature view."""
    if view not in VIEWS:
        raise ValueError(view)
    return np.hstack([blocks[k] for k in VIEWS[view]])


if __name__ == "__main__":
    df = load_dataset()
    blocks = build_feature_blocks(df, verbose=True)
    for k, v in blocks.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
    print("term classes:", pd.Series(blocks["classes"]).value_counts().to_dict())
    X = fusion_matrix(blocks, "B")
    print("fusion B:", X.shape, "finite:", np.isfinite(X).all())
