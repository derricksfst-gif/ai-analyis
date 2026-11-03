# -*- coding: utf-8 -*-
"""Bundle preprocessing + models + thresholds + applicability domain into one joblib."""
import os
import sys

import joblib

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, ROOT)
from src.final_pipeline import FinalPredictor  # noqa: E402


if __name__ == "__main__":
    predictor = FinalPredictor.load()
    out = os.path.join(ROOT, "models", "final_model.joblib")
    joblib.dump(predictor, out, compress=3)
    print("saved", out, f"({os.path.getsize(out)/1e6:.2f} MB)")
