# -*- coding: utf-8 -*-
"""Batch prediction with the end-to-end final_model.joblib.

Input: xlsx/csv with name,SMILES (or SMILES only).
Output: pred_value, class probabilities, nearest-neighbour similarity, AD status.
"""
import argparse
import os
import sys

import joblib

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, ROOT)
from src.final_pipeline import FinalPredictor  # noqa: F401,E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", default=os.path.join(ROOT, "predictions.csv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "final_model.joblib"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="classification probability threshold; default from model metadata")
    args = ap.parse_args()

    predictor = joblib.load(args.model)
    out = predictor.predict_file(args.input, args.output,
                                 probability_threshold=args.threshold)
    print(f"\nSaved {len(out)} predictions -> {args.output}")
    print(out.to_string(index=False))
    print(f"\nSupported gamma-Glu peptides: {int(out['pred_value'].notna().sum())}/{len(out)}")
    print("ad_status=in/borderline/out from Tanimoto + ChemBERTa kNN rules.")


if __name__ == "__main__":
    main()
