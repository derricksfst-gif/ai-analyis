# -*- coding: utf-8 -*-
"""5x5 stratified CV for the final task-specific models and classic benchmarks."""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M
import model_zoo as C

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = [42, 43, 44, 45, 46]
FINAL_XGB = dict(n_estimators=700, learning_rate=0.016455798888416407,
                 max_depth=3, min_child_weight=2.227504158038865,
                 subsample=0.876154068271453,
                 colsample_bytree=0.6920445674090548,
                 reg_lambda=0.4849080951384834,
                 reg_alpha=0.0020204294934293653, random_state=0,
                 n_jobs=4, verbosity=0)
FINAL_LGBM = dict(n_estimators=400, learning_rate=0.019347790880726648,
                  num_leaves=15, min_child_samples=12,
                  subsample=0.8453092258663428, subsample_freq=1,
                  colsample_bytree=0.5723428106589286,
                  reg_lambda=2.9686056115476545, random_state=0,
                  verbose=-1, n_jobs=4)


def main():
    df = M.load_dataset()
    y = df["value"].values
    y_cls = (y >= 6.0).astype(int)
    blocks = M.build_feature_blocks(df, need_embeddings=("cb",), verbose=True)
    X = M.fusion_matrix(blocks, "B_count")
    n = len(y)
    bins = pd.qcut(y, 5, labels=False, duplicates="drop")
    reg = C.reg_models()
    clf = C.clf_models()
    reg["XGBoost"] = xgb.XGBRegressor(**FINAL_XGB)
    reg["LightGBM"] = lgb.LGBMRegressor(**FINAL_LGBM)

    ro = {k: np.zeros((len(SEEDS), n)) for k in reg}
    co = {k: np.zeros((len(SEEDS), n)) for k in clf}
    t0 = time.time()

    for si, seed in enumerate(SEEDS):
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=seed).split(X, bins):
            sel = M.GainSelect(300).fit(X[tr], y[tr])
            Xtr, Xte = sel.transform(X[tr]), sel.transform(X[te])
            for name, model in reg.items():
                m = clone(model).fit(Xtr, y[tr])
                ro[name][si, te] = np.asarray(m.predict(Xte)).ravel()
        print(f"reg seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=seed).split(X, y_cls):
            sel = M.ETImportanceSelect(300).fit(X[tr], y_cls[tr])
            Xtr, Xte = sel.transform(X[tr]), sel.transform(X[te])
            for name, model in clf.items():
                m = clone(model).fit(Xtr, y_cls[tr])
                co[name][si, te] = m.predict_proba(Xte)[:, 1]
        print(f"clf seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    reg_rows, clf_rows = [], []
    for name, arr in ro.items():
        r2 = [r2_score(y, p) for p in arr]
        reg_rows.append(dict(
            model=name, R2=float(np.mean(r2)), R2_std=float(np.std(r2)),
            RMSE=float(np.mean([np.sqrt(mean_squared_error(y, p)) for p in arr])),
            MAE=float(np.mean([mean_absolute_error(y, p) for p in arr]))))
    for name, arr in co.items():
        aucs, accs, f1s, mccs, aps = [], [], [], [], []
        for p in arr:
            pred = (p >= 0.5).astype(int)
            aucs.append(roc_auc_score(y_cls, p))
            accs.append(accuracy_score(y_cls, pred))
            f1s.append(f1_score(y_cls, pred))
            mccs.append(matthews_corrcoef(y_cls, pred))
            aps.append(average_precision_score(y_cls, p))
        clf_rows.append(dict(model=name, AUC=float(np.mean(aucs)),
                             AUC_std=float(np.std(aucs)), AP=float(np.mean(aps)),
                             ACC=float(np.mean(accs)), F1=float(np.mean(f1s)),
                             MCC=float(np.mean(mccs))))
    reg_tab = pd.DataFrame(reg_rows).sort_values("R2", ascending=False)
    clf_tab = pd.DataFrame(clf_rows).sort_values("AUC", ascending=False)
    reg_tab.to_csv(os.path.join(OUT, "final_5x5_regression_benchmark.csv"),
                   index=False, encoding="utf-8-sig")
    clf_tab.to_csv(os.path.join(OUT, "final_5x5_classification_benchmark.csv"),
                   index=False, encoding="utf-8-sig")
    np.savez(os.path.join(OUT, "final_5x5_oof.npz"), y=y, y_cls=y_cls,
             **{f"reg_{k}": v for k, v in ro.items()},
             **{f"clf_{k}": v for k, v in co.items()})
    print("\n=== regression 5x5 ===")
    print(reg_tab.round(3).to_string(index=False))
    print("\n=== classification 5x5 ===")
    print(clf_tab.round(3).to_string(index=False))

    xgb_row = reg_tab[reg_tab.model == "XGBoost"].iloc[0]
    et = clf_tab[clf_tab.model == "ExtraTrees"].iloc[0]
    metrics = {
        "regression_5x5": {"model": "XGBoost", "R2_mean": float(xgb_row.R2),
                           "R2_std": float(xgb_row.R2_std), "RMSE": float(xgb_row.RMSE),
                           "MAE": float(xgb_row.MAE)},
        "classification_5x5": {"model": "ExtraTrees", "AUC": float(et.AUC),
                               "AUC_std": float(et.AUC_std), "AP": float(et.AP),
                               "ACC": float(et.ACC), "F1": float(et.F1),
                               "MCC": float(et.MCC)},
    }
    with open(os.path.join(OUT, "final_5x5_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "final_5x5_benchmark_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({"regression": reg_tab.to_dict(orient="records"),
                   "classification": clf_tab.to_dict(orient="records")},
                  f, indent=2, ensure_ascii=False)
    print("\nfinal:", metrics)


if __name__ == "__main__":
    main()
