# -*- coding: utf-8 -*-
"""Model zoo for the final classic-model benchmark."""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling as M

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEVICE = "cpu"


def scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def reg_models():
    return {
        "Ridge": scaled(Ridge(alpha=100.0)),
        "Lasso": scaled(Lasso(alpha=0.01, max_iter=50000)),
        "ElasticNet": scaled(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000)),
        "PLS(5)": scaled(PLSRegression(n_components=5)),
        "KNN(7)": scaled(KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "SVR-RBF": scaled(SVR(C=10.0, gamma="scale", epsilon=0.1)),
        "KRR-RBF": scaled(KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3)),
        "GP-Matern": scaled(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=5.0, nu=2.5)
            + WhiteKernel(0.1), normalize_y=True, optimizer=None,
            random_state=0)),
        "MLP": scaled(MLPRegressor(hidden_layer_sizes=(64,), alpha=10.0,
                                   learning_rate_init=1e-3, max_iter=1200,
                                   random_state=0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=600, min_samples_leaf=2, max_features="sqrt",
            n_jobs=4, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=2, max_features=0.3,
            n_jobs=4, random_state=0),
        "GBR": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=2,
            subsample=0.8, random_state=0),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=0),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.02, max_depth=3, subsample=0.85,
            colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
            n_jobs=4, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.02, num_leaves=7,
            min_child_samples=10, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
            verbose=-1, n_jobs=4),
        "CatBoost": CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=2, l2_leaf_reg=3.0,
            random_seed=0, verbose=False, allow_writing_files=False,
            thread_count=4),
    }


def clf_models():
    return {
        "LogReg": scaled(LogisticRegression(
            C=0.1, max_iter=10000, class_weight="balanced")),
        "LDA": scaled(LinearDiscriminantAnalysis()),
        "GaussianNB": scaled(GaussianNB()),
        "KNN(7)": scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVC-RBF": scaled(SVC(C=10.0, gamma="scale", probability=True,
                              class_weight="balanced", random_state=0)),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(64,), alpha=10.0,
                                    learning_rate_init=1e-3, max_iter=1200,
                                    random_state=0)),
        "RandomForest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=4, random_state=0),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=2, max_features=0.3,
            class_weight="balanced", n_jobs=4, random_state=0),
        "GBR": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=2,
            subsample=0.8, random_state=0),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=0),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=3, subsample=0.85,
            colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
            n_jobs=4, verbosity=0, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=7,
            min_child_samples=10, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
            verbose=-1, n_jobs=4),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=2, l2_leaf_reg=3.0,
            random_seed=0, verbose=False, allow_writing_files=False,
            thread_count=4),
    }


