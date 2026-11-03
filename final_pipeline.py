# -*- coding: utf-8 -*-
"""End-to-end predictor: features -> selected model -> regression/classification/AD."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import pairwise_distances

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(HERE) == "src":
    SRC = HERE
    ROOT = os.path.dirname(HERE)
else:
    SRC = HERE
    ROOT = HERE
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class FinalPredictor:
    def __init__(self, reg_bundle, clf_bundle, ad_reference=None, version="1.0"):
        self.reg_bundle = reg_bundle
        self.clf_bundle = clf_bundle
        self.ad_reference = ad_reference
        self.version = version

    @classmethod
    def load(cls, models_dir=None):
        models_dir = models_dir or os.path.join(ROOT, "models")
        reg = joblib.load(os.path.join(models_dir, "regression_xgboost.joblib"))
        clf = joblib.load(os.path.join(models_dir, "classification_extratrees.joblib"))
        ad_path = os.path.join(models_dir, "ad_reference.npz")
        ad = dict(np.load(ad_path)) if os.path.exists(ad_path) else None
        return cls(reg, clf, ad)

    def predict_dataframe(self, df, probability_threshold=None):
        import modeling as M

        df = df.copy()
        if "mol" not in df.columns:
            df["mol"] = [
                Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None
                for s in df["smiles"]
            ]
        n = len(df)
        core_ok = np.array([
            m is not None and m.HasSubstructMatch(M.CORE) for m in df["mol"]
        ])

        value = np.full(n, np.nan)
        prob = np.full(n, np.nan)
        if core_ok.any():
            sub = df.loc[core_ok].reset_index(drop=True)
            blocks = M.build_feature_blocks(sub, need_embeddings=("cb",),
                                            verbose=False)
            X = M.fusion_matrix(blocks, "B_count")
            Xr = X[:, self.reg_bundle["selector_idx"]]
            value[core_ok] = np.average(
                [m.predict(Xr) for m in self.reg_bundle["models"]],
                axis=0, weights=self.reg_bundle["weights"])
            value[core_ok] = np.clip(value[core_ok],
                                     *self.reg_bundle["metadata"]["clip"])
            prob[core_ok] = self.clf_bundle["model"].predict_proba(
                X[:, self.clf_bundle["selector_idx"]])[:, 1]

        med = self.reg_bundle["metadata"]["regression_threshold"]
        clf_thr = (probability_threshold if probability_threshold is not None
                   else self.clf_bundle["metadata"]["probability_threshold"])

        # Tanimoto
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        train = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
                          for s in self.reg_bundle["metadata"]["train_smiles"]],
                         dtype=np.float32)
        query = np.vstack([
            gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(2048, dtype=np.float32)
            for m in df["mol"]
        ]).astype(np.float32)
        inter = query @ train.T
        union = query.sum(1)[:, None] + train.sum(1)[None, :] - inter
        nn = (inter / np.maximum(union, 1e-6)).max(1)

        ad_knn = np.full(n, np.nan)
        ad_status = np.array(["unknown"] * n, dtype=object)
        if self.ad_reference is not None:
            eq = M.smiles_embeddings(df["smiles"].tolist(), "cb")
            zq = (eq - self.ad_reference["mean"]) / (self.ad_reference["std"] + 1e-9)
            d_q = pairwise_distances(zq, self.ad_reference["z_train"]).min(1)
            tan_ok = nn >= float(self.ad_reference["tanimoto_threshold"])
            knn_ok = d_q <= float(self.ad_reference["knn_threshold"])
            ad_status = np.where(tan_ok & knn_ok, "in",
                                 np.where(tan_ok | knn_ok, "borderline", "out"))
            ad_knn = np.round(d_q, 3)
        ad_status[~core_ok] = "out"

        return pd.DataFrame({
            "name": df["name"],
            "smiles": df["smiles"],
            "pred_value": np.round(value, 4),
            "pred_class_reg": np.where(np.isnan(value), "",
                                       np.where(value >= med, "high", "low")),
            "pred_class_clf": np.where(np.isnan(prob), "",
                                       np.where(prob >= clf_thr, "high", "low")),
            "prob_high": np.round(prob, 4),
            "max_similarity": np.round(nn, 3),
            "knn_distance": ad_knn,
            "ad_status": ad_status,
        })

    def predict_file(self, path, output=None, probability_threshold=None):
        import modeling as M

        df = M.read_query(path)
        out = self.predict_dataframe(df, probability_threshold=probability_threshold)
        if output:
            out.to_csv(output, index=False, encoding="utf-8-sig")
        return out


if __name__ == "__main__":
    p = FinalPredictor.load()
    print("loaded FinalPredictor", p.version)
    print(p.predict_file(os.path.join(ROOT, "example_input.csv")).head())
