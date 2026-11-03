# -*- coding: utf-8 -*-
"""Final performance figures for independent 5x5-CV models."""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "results")
FIG = os.path.join(RESULT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 600


def feature_names(idx):
    blocks = M.build_feature_blocks(M.load_dataset(), need_embeddings=())
    resid = list(blocks["resid_cols"])
    term = list(blocks["term_cols"])
    out = []
    for i in idx:
        i = int(i)
        if i < 384:
            out.append(f"ChemBERTa dim {i}")
        elif i < 384 + 52:
            nm = resid[i - 384]
            out.append(nm.replace("S_", "side.").replace("T_", "term."))
        elif i < 384 + 52 + 8:
            out.append(f"terminal={term[i - 384 - 52]}")
        elif i < 384 + 52 + 8 + 1024:
            out.append(f"AtomPair bit {i - 384 - 52 - 8}")
        else:
            out.append(f"MorganCount bit {i - 384 - 52 - 8 - 1024}")
    return out


def block_color(name):
    if name.startswith("ChemBERTa"):
        return "#4C72B0"
    if name.startswith("AtomPair"):
        return "#DD8452"
    if name.startswith("side.") or name.startswith("term."):
        return "#55A868"
    if name.startswith("terminal="):
        return "#8172B3"
    return "#8C8C8C"


def plot_regression_r2():
    tab = pd.read_csv(os.path.join(RESULT, "final_5x5_regression_benchmark.csv"))
    tab = tab.sort_values("R2")
    colors = ["#DD8452" if m == "XGBoost" else
              "#55A868" if m == "GBR" else "#4C72B0" for m in tab.model]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.barh(tab.model, tab.R2, color=colors)
    for i, v in enumerate(tab.R2):
        ax.text(max(v, 0) + 0.006, i, f"{v:.3f}", va="center", fontsize=7)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlim(min(-0.05, tab.R2.min() - 0.03), 0.66)
    ax.set_xlabel("Regression $R^2$ (5x5 stratified CV)")
    ax.set_title("Regression models: 5x5 CV $R^2$")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "regression_r2_comparison.png"),
                bbox_inches="tight")
    plt.close(fig)


def _roc_pr_curves(arr, y):
    fpr_grid = np.linspace(0, 1, 200)
    rec_grid = np.linspace(0, 1, 200)
    tprs, precs, aucs, aps = [], [], [], []
    for p in arr:
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(roc_auc_score(y, p))
        pr, rc, _ = precision_recall_curve(y, p)
        o = np.argsort(rc)
        precs.append(np.interp(rec_grid, rc[o], pr[o]))
        aps.append(average_precision_score(y, p))
    return (fpr_grid, rec_grid, np.mean(tprs, axis=0), np.mean(precs, axis=0),
            float(np.mean(aucs)), float(np.mean(aps)))


def plot_roc_pr():
    z = np.load(os.path.join(RESULT, "final_5x5_oof.npz"))
    y = z["y_cls"]
    keys = [k for k in z.files if k.startswith("clf_")]
    order = sorted(keys, key=lambda k: -np.mean(
        [roc_auc_score(y, p) for p in z[k]]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(order)))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for k, col in zip(order, colors):
        name = k.replace("clf_", "")
        fpr_grid, _, tpr, _, auc, _ = _roc_pr_curves(z[k], y)
        lw = 2.6 if name == "ExtraTrees" else 1.2
        alpha = 1.0 if name == "ExtraTrees" else 0.65
        ax.plot(fpr_grid, tpr, color=col, lw=lw, alpha=alpha,
                label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: 5x5 CV (mean of 5 seeds)")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "classification_roc_comparison.png"),
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for k, col in zip(order, colors):
        name = k.replace("clf_", "")
        _, rec_grid, _, prec, _, ap = _roc_pr_curves(z[k], y)
        lw = 2.6 if name == "ExtraTrees" else 1.2
        alpha = 1.0 if name == "ExtraTrees" else 0.65
        ax.plot(rec_grid, prec, color=col, lw=lw, alpha=alpha,
                label=f"{name} ({ap:.3f})")
    prev = float(y.mean())
    ax.axhline(prev, ls="--", color="0.5", lw=1)
    ax.text(0.99, prev + 0.025,
            f"At recall=1: precision = positive rate = {prev:.3f}",
            ha="right", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves: 5x5 CV (mean of 5 seeds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "classification_pr_comparison.png"),
                bbox_inches="tight")
    plt.close(fig)


def _sum_cm(arr, y_cls):
    cms = [confusion_matrix(y_cls, (p >= 0.5).astype(int)) for p in arr]
    cm = np.sum(cms, axis=0)
    acc = np.mean([accuracy_score(y_cls, (p >= 0.5).astype(int)) for p in arr])
    f1 = np.mean([f1_score(y_cls, (p >= 0.5).astype(int)) for p in arr])
    mcc = np.mean([matthews_corrcoef(y_cls, (p >= 0.5).astype(int)) for p in arr])
    return cm, acc, f1, mcc


def plot_confusion():
    z = np.load(os.path.join(RESULT, "final_5x5_oof.npz"))
    y_cls = z["y_cls"]
    arr = z["clf_ExtraTrees"]
    cm, acc, f1, mcc = _sum_cm(arr, y_cls)
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center",
                    va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Final classifier: ExtraTrees (5x5 CV, 5 seeds pooled)")
    ax.text(1.02, -0.20,
            f"ACC={acc:.3f}\nF1={f1:.3f}\nMCC={mcc:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_best_clf.png"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_compare():
    z = np.load(os.path.join(RESULT, "final_5x5_oof.npz"))
    y_cls = z["y_cls"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, name, note in [
        (axes[0], "ExtraTrees", "final classifier"),
        (axes[1], "HistGB", "comparison"),
    ]:
        arr = z["clf_" + name]
        cm, acc, f1, mcc = _sum_cm(arr, y_cls)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                pct = 100.0 * cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center",
                        va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_yticklabels(["Low", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(f"{name}\n({note})", fontsize=10)
        ax.text(0.02, -0.30,
                f"ACC={acc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}",
                transform=ax.transAxes, fontsize=7.5, va="top")
    fig.suptitle("Confusion matrices at threshold 0.5 (5x5 CV, 5 seeds pooled)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_et_vs_histgb.png"),
                bbox_inches="tight")
    plt.close(fig)


def plot_confusion_histgb():
    z = np.load(os.path.join(RESULT, "final_5x5_oof.npz"))
    y_cls = z["y_cls"]
    arr = z["clf_HistGB"]
    cm, acc, f1, mcc = _sum_cm(arr, y_cls)
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center",
                    va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("HistGB: confusion matrix (5x5 CV)")
    ax.text(1.02, -0.20,
            f"ACC={acc:.3f}\nF1={f1:.3f}\nMCC={mcc:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_histgb.png"), bbox_inches="tight")
    plt.close(fig)


def plot_shap():
    bundle = joblib.load(os.path.join(ROOT, "models", "classification_extratrees.joblib"))
    df = M.load_dataset()
    blocks = M.build_feature_blocks(df, need_embeddings=("cb",))
    X = M.fusion_matrix(blocks, "B_count")[:, bundle["selector_idx"]]
    names = feature_names(bundle["selector_idx"])
    expl = shap.TreeExplainer(bundle["model"])
    vals = expl.shap_values(X)
    sv = vals[:, :, 1] if np.ndim(vals) == 3 else vals
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)

    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.barh(range(len(top)), mean_abs[top],
            color=[block_color(names[j]) for j in top])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([names[j] for j in top], fontsize=7)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP summary: ExtraTrees classification (selected features)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shap_clf_bar.png"), bbox_inches="tight")
    plt.close(fig)

    top2 = order[:15][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    rng = np.random.RandomState(0)
    norm = Normalize(vmin=0, vmax=1)
    for rank, j in enumerate(top2):
        vals_j = X[:, j]
        lo, hi = float(vals_j.min()), float(vals_j.max())
        c = (vals_j - lo) / max(hi - lo, 1e-9)
        ypos = rank + rng.uniform(-0.16, 0.16, len(vals_j))
        ax.scatter(sv[:, j], ypos, c=c, cmap="coolwarm", norm=norm,
                   s=16, alpha=0.85, linewidths=0)
    ax.set_yticks(range(len(top2)))
    ax.set_yticklabels([names[j] for j in top2], fontsize=7)
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("SHAP detail: ExtraTrees classification")
    sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02,
                 label="Feature value (low to high)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shap_clf_beeswarm.png"), bbox_inches="tight")
    plt.close(fig)
    print("top SHAP features:", [names[j] for j in order[:8]])


def plot_shap_regression():
    bundle = joblib.load(os.path.join(ROOT, "models", "regression_xgboost.joblib"))
    df = M.load_dataset()
    blocks = M.build_feature_blocks(df, need_embeddings=("cb",))
    X = M.fusion_matrix(blocks, "B_count")[:, bundle["selector_idx"]]
    names = feature_names(bundle["selector_idx"])
    booster = bundle["models"][0].get_booster()
    sv = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)

    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.barh(range(len(top)), mean_abs[top],
            color=[block_color(names[j]) for j in top])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([names[j] for j in top], fontsize=7)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP summary: XGBoost regression (selected features)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shap_reg_bar.png"), bbox_inches="tight")
    plt.close(fig)

    top2 = order[:15][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    rng = np.random.RandomState(0)
    norm = Normalize(vmin=0, vmax=1)
    for rank, j in enumerate(top2):
        vals_j = X[:, j]
        lo, hi = float(vals_j.min()), float(vals_j.max())
        c = (vals_j - lo) / max(hi - lo, 1e-9)
        ypos = rank + rng.uniform(-0.16, 0.16, len(vals_j))
        ax.scatter(sv[:, j], ypos, c=c, cmap="coolwarm", norm=norm,
                   s=16, alpha=0.85, linewidths=0)
    ax.set_yticks(range(len(top2)))
    ax.set_yticklabels([names[j] for j in top2], fontsize=7)
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("SHAP detail: XGBoost regression")
    sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02,
                 label="Feature value (low to high)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shap_reg_scatter.png"), bbox_inches="tight")
    plt.close(fig)
    print("reg top SHAP:", [names[j] for j in order[:8]])


def main():
    plot_regression_r2()
    plot_roc_pr()
    plot_confusion()
    plot_confusion_compare()
    plot_confusion_histgb()
    plot_shap_regression()
    plot_shap()
    print("final figures saved")


if __name__ == "__main__":
    main()
