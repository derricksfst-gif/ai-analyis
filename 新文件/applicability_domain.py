# -*- coding: utf-8 -*-
"""Applicability domain: t-SNE visualization + Tanimoto + ChemBERTa kNN distance."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "results")
FIG = os.path.join(RESULT, "figures")
MODELS = os.path.join(ROOT, "models")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 600

TANIMOTO_THRESHOLD = 0.50
KNN_QUANTILE = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--new", required=True,
                    help="new molecules xlsx/csv: name,SMILES (or SMILES only)")
    ap.add_argument("-o", "--output", default=os.path.join(RESULT, "applicability_domain.csv"))
    ap.add_argument("--figure", default=os.path.join(FIG, "applicability_domain_tsne.png"))
    ap.add_argument("--tanimoto-threshold", type=float, default=TANIMOTO_THRESHOLD)
    ap.add_argument("--knn-quantile", type=float, default=KNN_QUANTILE)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.figure)), exist_ok=True)

    train = M.load_dataset()
    query = M.read_query(args.new)
    etr = M.smiles_embeddings(train["smiles"].tolist(), "cb")
    eq = M.smiles_embeddings(query["smiles"].tolist(), "cb")

    mu = etr.mean(0)
    sd = etr.std(0) + 1e-9
    ztr = (etr - mu) / sd
    zq = (eq - mu) / sd

    # kNN distance in ChemBERTa space
    d_tr = pairwise_distances(ztr, ztr)
    np.fill_diagonal(d_tr, np.inf)
    nn_tr = d_tr.min(1)
    knn_thr = float(np.quantile(nn_tr, args.knn_quantile))
    d_q = pairwise_distances(zq, ztr).min(1)

    # Tanimoto to training
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ftr = np.array([gen.GetFingerprintAsNumPy(m) for m in train["mol"]],
                   dtype=np.float32)
    fq = np.array([gen.GetFingerprintAsNumPy(m) for m in query["mol"]],
                  dtype=np.float32)
    inter = fq @ ftr.T
    union = fq.sum(1)[:, None] + ftr.sum(1)[None, :] - inter
    tan = (inter / np.maximum(union, 1e-6)).max(1)

    # t-SNE on combined space (visualization only)
    zall = np.vstack([ztr, zq])
    tsne = TSNE(n_components=2, perplexity=15, learning_rate="auto",
                init="pca", random_state=42)
    y = tsne.fit_transform(zall)
    ytr, yq = y[:len(train)], y[len(train):]
    hull = ConvexHull(ytr)
    hull_path = Path(ytr[hull.vertices])
    inside = hull_path.contains_points(yq)

    tan_ok = tan >= args.tanimoto_threshold
    knn_ok = d_q <= knn_thr
    status = np.where(tan_ok & knn_ok, "in",
                      np.where(tan_ok | knn_ok, "borderline", "out"))

    out = pd.DataFrame({
        "name": query["name"],
        "smiles": query["smiles"],
        "max_similarity": np.round(tan, 3),
        "knn_distance": np.round(d_q, 3),
        "knn_threshold": round(knn_thr, 3),
        "tsne_inside_hull": inside,
        "ad_status": status,
    })
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    np.savez_compressed(os.path.join(MODELS, "ad_reference.npz"),
                        z_train=ztr, mean=mu, std=sd,
                        knn_threshold=knn_thr,
                        tanimoto_threshold=args.tanimoto_threshold)

    # plot
    colors = {"in": "#2E8B57", "borderline": "#E69F00", "out": "#C0392B"}
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(ytr[:, 0], ytr[:, 1], s=42, color="#8C8C8C", alpha=0.75,
               label="Training (n=60)", edgecolor="white", linewidth=0.4)
    hv = ytr[hull.vertices]
    ax.plot(np.append(hv[:, 0], hv[0, 0]), np.append(hv[:, 1], hv[0, 1]),
            "--", color="#4C72B0", lw=1.2, label="Training convex hull")
    for st in ["in", "borderline", "out"]:
        m = status == st
        if m.any():
            ax.scatter(yq[m, 0], yq[m, 1], s=68, color=colors[st],
                       edgecolor="black", linewidth=0.5, zorder=4,
                       label=f"New: {st}")
            for i in np.where(m)[0]:
                ax.annotate(query["name"].iloc[i], (yq[i, 0], yq[i, 1]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Applicability domain: t-SNE of ChemBERTa embeddings")
    ax.text(0.02, 0.02,
            f"AD decision = Tanimoto >= {TANIMOTO_THRESHOLD:.2f}\n"
            f"AND kNN distance <= {knn_thr:.3f} (train 95th pct)\n"
            "Hull is visual only",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
