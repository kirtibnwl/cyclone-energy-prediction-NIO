"""Evaluation utilities: metrics, permutation importance, feature selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Metrics:
    mse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        return f"MSE={self.mse:.6f}  MAE={self.mae:.6f}  R2={self.r2:.4f}"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mse=float(mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def compute_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation feature importance, sorted descending."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state
    )
    df = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    )
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def select_top_features(importance_df: pd.DataFrame, threshold: float = 0.0) -> list[str]:
    """Return features whose mean importance is above ``threshold``."""
    return importance_df.loc[importance_df["Importance"] > threshold, "Feature"].tolist()
