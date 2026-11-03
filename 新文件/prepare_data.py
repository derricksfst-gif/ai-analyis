# -*- coding: utf-8 -*-
"""Build the cleaned dataset with pEC50, derived EC50, and class labels."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


def main():
    df = M.load_dataset()
    p = df["value"].values
    med = 6.0
    out = pd.DataFrame({
        "name": df["name"],
        "smiles": df["smiles"],
        "pEC50": p,
        "EC50_M": 10.0 ** (-p),
        "EC50_uM": 10.0 ** (-p) * 1e6,
        "class_label": np.where(p >= med, "high", "low"),
    })
    csv = os.path.join(OUT, "dataset.csv")
    xlsx = os.path.join(OUT, "dataset.xlsx")
    out.to_csv(csv, index=False, encoding="utf-8-sig")
    out.to_excel(xlsx, index=False)
    print(f"saved {csv} ({len(out)} rows)")
    print(f"classification threshold = pEC50 >= {med:.1f}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
