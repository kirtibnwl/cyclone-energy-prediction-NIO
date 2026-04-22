"""Smoke tests for the cyclone-energy-prediction pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``src`` importable when pytest is invoked from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_clean, split_features_target  # noqa: E402
from src.evaluate import compute_feature_importance, compute_metrics  # noqa: E402
from src.models import build_ann  # noqa: E402


def test_clean_dataset_has_expected_columns():
    df = load_clean("data/cyclone_data_clean.csv")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "Year"]:
        assert col in df.columns
    assert len(df) >= 40


def test_split_features_target_shapes():
    df = load_clean("data/cyclone_data_clean.csv")
    X, y = split_features_target(df)
    assert X.shape[0] == y.shape[0]
    assert list(X.columns) == FEATURE_COLUMNS


def test_model_fits_and_predicts():
    df = load_clean("data/cyclone_data_clean.csv")
    X, y = split_features_target(df)

    model = build_ann(max_iter=200)
    # Normalise inputs so the MLP converges quickly in the test.
    X_scaled = (X - X.mean()) / X.std()
    y_scaled = (y - y.mean()) / y.std()

    model.fit(X_scaled.values, y_scaled.values)
    preds = model.predict(X_scaled.values)
    assert preds.shape == y_scaled.shape

    metrics = compute_metrics(y_scaled.values, preds)
    assert metrics.r2 > 0.5  # Sanity: model should explain some variance


def test_permutation_importance_runs():
    df = load_clean("data/cyclone_data_clean.csv")
    X, y = split_features_target(df)
    X_scaled = ((X - X.mean()) / X.std()).values
    y_scaled = ((y - y.mean()) / y.std()).values

    model = build_ann(max_iter=200)
    model.fit(X_scaled, y_scaled)

    importance = compute_feature_importance(model, X_scaled, y_scaled, FEATURE_COLUMNS, n_repeats=3)
    assert set(importance["Feature"]) == set(FEATURE_COLUMNS)
