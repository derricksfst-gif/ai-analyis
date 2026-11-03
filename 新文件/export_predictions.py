# -*- coding: utf-8 -*-
"""Export every training molecule's 5x5 OOF prediction, error and class score."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def main():
    df = M.load_dataset()
    z = np.load(os.path.join(OUT, "final_5x5_oof.npz"))
    y = z["y"]
    y_cls = z["y_cls"]
    reg = z["reg_XGBoost"]
    clf = z["clf_ExtraTrees"]
    pred = reg.mean(0)
    pred_std = reg.std(0)
    prob = clf.mean(0)
    resid = y - pred
    pred_cls = (prob >= 0.5).astype(int)
    out = pd.DataFrame({
        "index": np.arange(1, len(df) + 1),
        "name": df["name"],
        "smiles": df["smiles"],
        "pEC50_true": y,
        "pEC50_oof": np.round(pred, 4),
        "oof_std_across_5x5": np.round(pred_std, 4),
        "residual": np.round(resid, 4),
        "abs_error": np.round(np.abs(resid), 4),
        "class_true": np.where(y_cls == 1, "high", "low"),
        "prob_high_oof": np.round(prob, 4),
        "class_pred": np.where(pred_cls == 1, "high", "low"),
        "class_correct": np.where(pred_cls == y_cls, "yes", "no"),
    })
    out.to_csv(os.path.join(OUT, "all_predictions_5x5.csv"), index=False,
               encoding="utf-8-sig")
    print(out.head(10).to_string(index=False))
    print(f"\nsaved {len(out)} rows -> results/all_predictions_5x5.csv")
    print("mean abs error:", round(np.abs(resid).mean(), 3),
          "max abs error:", round(np.abs(resid).max(), 3))


if __name__ == "__main__":
    main()
