"""Plotting helpers. All figures are saved to the ``results/`` directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _save(fig, name: str) -> Path:
    out = RESULTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_true, label="Actual NIO_ACE", marker="o", color="#1f77b4")
    ax.plot(y_pred, label="Predicted NIO_ACE", marker="x", color="#ff7f0e", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("NIO_ACE (10^4 knots^2)")
    ax.legend()
    return _save(fig, filename)


def plot_training_loss(loss_curve: list[float], filename: str = "training_loss.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_curve, color="#2ca02c")
    ax.set_title("ANN Training Loss Curve")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss (MSE on scaled target)")
    return _save(fig, filename)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residuals.png") -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(residuals, marker="o", color="#2ca02c")
    ax.axhline(0, color="red", linestyle="--", label="Zero Error")
    ax.set_title("Residuals (Actual - Predicted)")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Residual")
    ax.legend()
    return _save(fig, filename)


def plot_residual_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residual_hist.png"
) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=12, kde=True, color="purple", alpha=0.7, ax=ax)
    ax.set_title("Residual Error Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    return _save(fig, filename)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "scatter.png") -> Path:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, s=80, color="#1f77b4", edgecolor="w", ax=ax)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect Fit")
    ax.set_title("Actual vs Predicted NIO_ACE")
    ax.set_xlabel("Actual NIO_ACE")
    ax.set_ylabel("Predicted NIO_ACE")
    ax.legend()
    return _save(fig, filename)


def plot_feature_importance(importance_df: pd.DataFrame, filename: str = "feature_importance.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Importance via Permutation")
    ax.set_xlabel("Mean Decrease in Model Performance")
    return _save(fig, filename)


def plot_pairplot(df: pd.DataFrame, cols: list[str], filename: str = "pairplot.png") -> Path:
    g = sns.pairplot(df[cols], diag_kind="kde", corner=True)
    g.fig.suptitle("Pair Plot of NIO Cyclone Metrics", y=1.02)
    out = RESULTS_DIR / filename
    g.fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return out


def plot_model_comparison(
    y_true: np.ndarray,
    y_initial: np.ndarray,
    y_optimized: np.ndarray,
    filename: str = "model_comparison.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label="Actual ACE", marker="o", color="#1f77b4")
    ax.plot(y_initial, label="Initial ANN", marker="^", linestyle="-.", color="#2ca02c")
    ax.plot(y_optimized, label="Optimized ANN", marker="x", linestyle="--", color="#ff7f0e")
    ax.set_title("Actual vs Initial ANN vs Optimized ANN")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("NIO_ACE (10^4 knots^2)")
    ax.legend()
    return _save(fig, filename)
