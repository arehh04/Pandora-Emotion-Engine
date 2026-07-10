"""Regression and tiered-classification metrics for comparing the LLM agent
against the historical classical-ML pipeline (see Task 4's hardcoded
HISTORICAL_BASELINES for the classical side).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_tier_metrics(tier_true, tier_pred):
    labels = sorted(set(tier_true) | set(tier_pred))
    accuracy = float(accuracy_score(tier_true, tier_pred))
    macro_f1 = float(f1_score(tier_true, tier_pred, average="macro", labels=labels, zero_division=0))
    weighted_kappa = float(cohen_kappa_score(tier_true, tier_pred, weights="linear", labels=labels))
    cm = confusion_matrix(tier_true, tier_pred, labels=labels).tolist()
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_kappa": weighted_kappa,
        "confusion_matrix": cm,
    }
