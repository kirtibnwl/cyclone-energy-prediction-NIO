"""End-to-end training pipeline: initial ANN -> permutation importance -> optimised ANN.

Reproduces the workflow described in Beniwal & Kumar (2026):

    1. Train an initial MLP with all six features.
    2. Compute permutation feature importance on the scaled test set.
    3. Drop non-contributing features and retrain.
    4. Compare initial vs optimised metrics and save all figures.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_clean,
    split_features_target,
)
from .evaluate import (
    compute_feature_importance,
    compute_metrics,
    select_top_features,
)
from .models import build_ann
from .visualize import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_pairplot,
    plot_residual_distribution,
    plot_residuals,
    plot_scatter,
    plot_training_loss,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _fit_scaled_ann(X_train, X_test, y_train, y_test):
    """Scale inputs/target, fit an MLP, return (model, scaler_y, y_pred_orig, y_test_orig)."""
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    model = build_ann()
    model.fit(X_train_s, y_train_s)

    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).ravel()
    return model, scaler_X, scaler_y, X_test_s, y_test_s, y_pred, y_test_orig


def run(
    data_path: str | Path = "data/cyclone_data_clean.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Run the full pipeline and return a summary dictionary."""
    print("Loading cleaned dataset…")
    df = load_clean(data_path)
    X, y = split_features_target(df, target=TARGET_COLUMN, features=FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 1. Initial ANN on all 6 features
    # ------------------------------------------------------------------
    print("\n[1/3] Training initial ANN with all features…")
    (
        init_model,
        init_scaler_X,
        init_scaler_y,
        X_test_s_init,
        y_test_s_init,
        y_pred_init,
        y_test_orig,
    ) = _fit_scaled_ann(X_train, X_test, y_train, y_test)

    init_metrics = compute_metrics(y_test_orig, y_pred_init)
    print(f"  Initial model: {init_metrics}")

    plot_actual_vs_predicted(
        y_test_orig, y_pred_init, "Initial ANN: Actual vs Predicted NIO_ACE", "initial_actual_vs_predicted.png"
    )
    plot_training_loss(init_model.loss_curve_, "initial_training_loss.png")
    plot_residuals(y_test_orig, y_pred_init, "initial_residuals.png")
    plot_residual_distribution(y_test_orig, y_pred_init, "initial_residual_hist.png")
    plot_scatter(y_test_orig, y_pred_init, "initial_scatter.png")
    plot_pairplot(df, ["NIO_VF", "NIO_ACE", "NIO_PDI"], "nio_pairplot.png")

    # ------------------------------------------------------------------
    # 2. Permutation feature importance
    # ------------------------------------------------------------------
    print("\n[2/3] Computing permutation feature importance…")
    importance_df = compute_feature_importance(
        init_model, X_test_s_init, y_test_s_init, FEATURE_COLUMNS
    )
    print(importance_df)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df, "feature_importance.png")

    top_features = select_top_features(importance_df, threshold=0.0)
    if not top_features:
        # Guardrail: keep top 3 if no feature clears the zero threshold.
        top_features = importance_df.head(3)["Feature"].tolist()
    print(f"  Selected top features: {top_features}")

    # ------------------------------------------------------------------
    # 3. Optimised ANN using only top features
    # ------------------------------------------------------------------
    print("\n[3/3] Retraining optimized ANN with top features only…")
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    (
        opt_model,
        _opt_scaler_X,
        _opt_scaler_y,
        _X_test_s_opt,
        _y_test_s_opt,
        y_pred_opt,
        _y_test_orig_opt,
    ) = _fit_scaled_ann(X_train_top, X_test_top, y_train, y_test)

    opt_metrics = compute_metrics(y_test_orig, y_pred_opt)
    print(f"  Optimized model: {opt_metrics}")

    plot_actual_vs_predicted(
        y_test_orig, y_pred_opt, "Optimized ANN: Actual vs Predicted NIO_ACE", "optimized_actual_vs_predicted.png"
    )
    plot_training_loss(opt_model.loss_curve_, "optimized_training_loss.png")
    plot_model_comparison(y_test_orig, y_pred_init, y_pred_opt, "model_comparison.png")

    # ------------------------------------------------------------------
    # Persist a summary
    # ------------------------------------------------------------------
    summary = {
        "initial": asdict(init_metrics),
        "optimized": asdict(opt_metrics),
        "top_features": top_features,
        "mse_reduction_pct": 100.0 * (init_metrics.mse - opt_metrics.mse) / init_metrics.mse,
        "r2_delta": opt_metrics.r2 - init_metrics.r2,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save predictions for downstream use
    pd.DataFrame(
        {
            "y_actual": y_test_orig,
            "y_pred_initial": y_pred_init,
            "y_pred_optimized": y_pred_opt,
        }
    ).to_csv(RESULTS_DIR / "predictions.csv", index=False)

    print("\nDone. All outputs written to ./results/")
    return summary


if __name__ == "__main__":
    run()
