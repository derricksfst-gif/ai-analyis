# -*- coding: utf-8 -*-
"""Performance report: training, 5x5 CV, fixed 80/20 test, 20 repeated holdouts."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M
import xgboost as xgb

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
XGB = dict(n_estimators=700, learning_rate=0.016455798888416407,
           max_depth=3, min_child_weight=2.227504158038865,
           subsample=0.876154068271453, colsample_bytree=0.6920445674090548,
           reg_lambda=0.4849080951384834,
           reg_alpha=0.0020204294934293653,
           random_state=0, n_jobs=4, verbosity=0)


def fit_eval(tr, te, X, y, y_cls):
    Xtr, Xte = X[tr], X[te]
    # regression
    sr = M.GainSelect(300).fit(Xtr, y[tr])
    Xr, Xre = sr.transform(Xtr), sr.transform(Xte)
    mx = xgb.XGBRegressor(**XGB).fit(Xr, y[tr])
    ptr = mx.predict(Xr)
    pte = mx.predict(Xre)
    reg = dict(
        train_R2=r2_score(y[tr], ptr),
        train_RMSE=float(np.sqrt(mean_squared_error(y[tr], ptr))),
        train_MAE=mean_absolute_error(y[tr], ptr),
        test_R2=r2_score(y[te], pte),
        test_RMSE=float(np.sqrt(mean_squared_error(y[te], pte))),
        test_MAE=mean_absolute_error(y[te], pte),
    )
    # classification
    sc = M.ETImportanceSelect(300).fit(Xtr, y_cls[tr])
    Xc, Xce = sc.transform(Xtr), sc.transform(Xte)
    mc = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                              max_features=0.3, class_weight="balanced",
                              n_jobs=4, random_state=0).fit(Xc, y_cls[tr])
    ptrc = mc.predict_proba(Xc)[:, 1]
    ptec = mc.predict_proba(Xce)[:, 1]

    def cm(p, yy):
        pred = (p >= 0.5).astype(int)
        return dict(AUC=roc_auc_score(yy, p), ACC=accuracy_score(yy, pred),
                    F1=f1_score(yy, pred), MCC=matthews_corrcoef(yy, pred))

    ctr, cte = cm(ptrc, y_cls[tr]), cm(ptec, y_cls[te])
    clf = {f"train_{k}": v for k, v in ctr.items()}
    clf.update({f"test_{k}": v for k, v in cte.items()})
    return reg, clf


def main():
    df = M.load_dataset()
    y = df["value"].values
    y_cls = (y >= 6.0).astype(int)
    X = M.fusion_matrix(M.build_feature_blocks(
        df, need_embeddings=("cb",), verbose=True), "B_count")
    bins = pd.qcut(y, 5, labels=False, duplicates="drop")

    # fixed 80/20 test
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(split.split(X, bins))
    reg42, clf42 = fit_eval(tr, te, X, y, y_cls)
    print("fixed 80/20 (seed 42):")
    print("  reg", {k: round(v, 3) for k, v in reg42.items()})
    print("  clf", {k: round(v, 3) for k, v in clf42.items()})

    # 20 repeated holdouts
    reg_rows, clf_rows = [], []
    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=123)
    for tr, te in sss.split(X, bins):
        r, c = fit_eval(tr, te, X, y, y_cls)
        reg_rows.append(r)
        clf_rows.append(c)
    reg_df = pd.DataFrame(reg_rows)
    clf_df = pd.DataFrame(clf_rows)
    print("20 holdouts mean:")
    print("  reg", reg_df.mean().round(3).to_dict())
    print("  clf", clf_df.mean().round(3).to_dict())

    with open(os.path.join(OUT, "final_5x5_metrics.json"),
              encoding="utf-8") as f:
        cv = json.load(f)
    rows = []
    for task, metrics in [
        ("regression", ["R2", "RMSE", "MAE"]),
        ("classification", ["AUC", "ACC", "F1", "MCC"]),
    ]:
        for met in metrics:
            if task == "regression":
                tr_val = reg_df[f"train_{met}"].mean()
                te42 = reg42[f"test_{met}"]
                te_mean = reg_df[f"test_{met}"].mean()
                te_std = reg_df[f"test_{met}"].std()
                cv_val = cv["regression_5x5"].get("R2_mean" if met == "R2" else met, np.nan)
            else:
                tr_val = clf_df[f"train_{met}"].mean()
                te42 = clf42[f"test_{met}"]
                te_mean = clf_df[f"test_{met}"].mean()
                te_std = clf_df[f"test_{met}"].std()
                cv_val = cv["classification_5x5"][met]
            rows.append(dict(task=task, metric=met, train=tr_val,
                             cv_5x5=cv_val, test_seed42=te42,
                             test_20_mean=te_mean, test_20_std=te_std))
    perf = pd.DataFrame(rows)
    perf.to_csv(os.path.join(OUT, "model_performance_train_cv_test.csv"),
                index=False, encoding="utf-8-sig")
    print("\n", perf.round(3).to_string(index=False))
    with open(os.path.join(OUT, "model_performance.json"), "w",
              encoding="utf-8") as f:
        json.dump({"fixed_seed42": {"regression": reg42, "classification": clf42},
                   "holdout20_mean": {"regression": reg_df.mean().to_dict(),
                                      "classification": clf_df.mean().to_dict()},
                   "holdout20_std": {"regression": reg_df.std().to_dict(),
                                     "classification": clf_df.std().to_dict()},
                   "cv_5x5": cv}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
