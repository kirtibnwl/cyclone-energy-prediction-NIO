"""Data loading and preprocessing for the IMD cyclone dataset.

The raw file ``data/Table_3.csv`` mirrors the layout of the original IMD
export used in Beniwal & Kumar (2026): two header rows precede the data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

COLUMNS = [
    "Year",
    "BOB_VF", "BOB_ACE", "BOB_PDI",
    "AS_VF",  "AS_ACE",  "AS_PDI",
    "NIO_VF", "NIO_ACE", "NIO_PDI",
]

FEATURE_COLUMNS = ["BOB_VF", "BOB_PDI", "AS_VF", "AS_PDI", "NIO_VF", "NIO_PDI"]
TARGET_COLUMN = "NIO_ACE"
ALL_TARGETS = ["BOB_ACE", "AS_ACE", "NIO_ACE"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw IMD export and normalise column names / dtypes."""
    path = Path(path)
    raw = pd.read_csv(path)

    # The first two rows of the raw export are a multi-index header; drop them.
    cleaned = raw.drop([0, 1]).reset_index(drop=True)
    cleaned.columns = COLUMNS
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def load_clean(path: str | Path = "data/cyclone_data_clean.csv") -> pd.DataFrame:
    """Load the already-cleaned CSV (preferred for reproducible runs)."""
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    features: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for modelling."""
    features = features or FEATURE_COLUMNS
    return df[features].copy(), df[target].copy()
