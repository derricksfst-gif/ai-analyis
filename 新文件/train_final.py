# -*- coding: utf-8 -*-
"""Train final independent models: LightGBM regression + ExtraTrees classification."""
import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

XGB = dict(n_estimators=700, learning_rate=0.016455798888416407,
           max_depth=3, min_child_weight=2.227504158038865,
           subsample=0.876154068271453, colsample_bytree=0.6920445674090548,
           reg_lambda=0.4849080951384834,
           reg_alpha=0.0020204294934293653,
           n_jobs=4, verbosity=0)


def main():
    df = M.load_dataset()
    y = df["value"].values
    y_cls = (y >= 6.0).astype(int)
    blocks = M.build_feature_blocks(df, need_embeddings=("cb",), verbose=True)
    X = M.fusion_matrix(blocks, "B_count")

    # regression: task-specific features, single XGBoost
    sel_r = M.GainSelect(300).fit(X, y)
    Xr = sel_r.transform(X)
    reg_models = [xgb.XGBRegressor(**XGB, random_state=0).fit(Xr, y)]
    reg_weights = [1.0]

    # classification: task-specific features
    sel_c = M.ETImportanceSelect(300).fit(X, y_cls)
    Xc = sel_c.transform(X)
    clf = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                               max_features=0.3, class_weight="balanced",
                               n_jobs=4, random_state=0).fit(Xc, y_cls)

    reg_train = np.average([m.predict(Xr) for m in reg_models], axis=0,
                           weights=reg_weights)
    print("in-sample R2:", round(r2_score(y, reg_train), 3),
          "in-sample ACC:", round(accuracy_score(y_cls, clf.predict(Xc)), 3))

    with open(os.path.join(RESULT, "final_5x5_metrics.json"),
              encoding="utf-8") as f:
        cv_metrics = json.load(f)
    reg_metrics = cv_metrics["regression_5x5"]
    clf_metrics = cv_metrics["classification_5x5"]
    common = {
        "train_names": df["name"].tolist(),
        "train_smiles": df["smiles"].tolist(),
        "regression_threshold": 6.0,
        "probability_threshold": 0.5,
        "clip": [float(y.min() - 0.5), float(y.max() + 0.5)],
    }
    reg_bundle = {
        "selector_idx": sel_r.idx_,
        "models": reg_models,
        "weights": reg_weights,
        "metrics": reg_metrics,
        "metadata": {**common, "task": "regression",
                     "feature": "GainSelect(300)",
                     "model": "XGBoost"},
    }
    clf_bundle = {
        "selector_idx": sel_c.idx_,
        "model": clf,
        "metrics": clf_metrics,
        "metadata": {**common, "task": "classification",
                     "feature": "ETImportanceSelect(300)"},
    }
    joblib.dump(reg_bundle, os.path.join(MODELS, "regression_xgboost.joblib"), compress=3)
    joblib.dump(clf_bundle, os.path.join(MODELS, "classification_extratrees.joblib"), compress=3)
    meta = {"regression": reg_bundle["metrics"],
            "classification": clf_bundle["metrics"],
            "regression_threshold": common["regression_threshold"],
            "probability_threshold": common["probability_threshold"],
            "n_features_reg": int(len(sel_r.idx_)),
            "n_features_clf": int(len(sel_c.idx_))}
    with open(os.path.join(MODELS, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("saved regression_xgboost.joblib and classification_extratrees.joblib")


if __name__ == "__main__":
    main()
