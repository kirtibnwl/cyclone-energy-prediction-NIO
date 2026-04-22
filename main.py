"""Command-line entry point for the cyclone-energy-prediction pipeline.

Usage:
    python main.py
    python main.py --data data/cyclone_data_clean.csv --test-size 0.2
"""
from __future__ import annotations

import argparse

from src.train import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ANN cyclone-energy model (Beniwal & Kumar, 2026)."
    )
    parser.add_argument(
        "--data",
        default="data/cyclone_data_clean.csv",
        help="Path to the cleaned cyclone CSV (default: data/cyclone_data_clean.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(data_path=args.data, test_size=args.test_size, random_state=args.seed)

    print("\n=== Summary ===")
    print(f"Initial   : MSE={summary['initial']['mse']:.6f}  R2={summary['initial']['r2']:.4f}")
    print(f"Optimized : MSE={summary['optimized']['mse']:.6f}  R2={summary['optimized']['r2']:.4f}")
    print(f"MSE reduction: {summary['mse_reduction_pct']:.2f}%")
    print(f"Selected features: {', '.join(summary['top_features'])}")


if __name__ == "__main__":
    main()
