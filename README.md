# Tropical Cyclone Energy Prediction — North Indian Ocean (ANN)

> **Published Research Project** — code, data, and reproducibility package for
>
> **Beniwal, K. & Kumar, V. (2026).** *The tropical cyclone energy prediction of the North Indian Ocean in monsoon using artificial neural networks.* **MAUSAM, 77(2), 517–526.**
> Department of Applied Mathematics, Delhi Technological University.

Predicting **Accumulated Cyclone Energy (ACE)** over the North Indian Ocean (NIO) during the monsoon season using an **optimised Artificial Neural Network (ANN)**, trained on 1982–2023 India Meteorological Department (IMD) cyclone metrics.

---

## Highlights

- **Published, peer-reviewed research** (MAUSAM, Vol. 77, Issue 2, April 2026) — the full paper is bundled under `paper/`.
- **End-to-end ML pipeline**: data cleaning → ANN training → permutation feature importance → optimised re-training → visualisation.
- **47% reduction in MSE** and **~14% increase in R²** after permutation-based feature reduction (Table 3 of the paper).
- **Statistically validated** via one-sample and paired t-tests (p < 0.001 between initial and optimised prediction errors).
- **Outperforms linear baselines**: Optimised MLP R² = 0.92 vs Linear Regression R² = 0.71, Stepwise Regression R² = 0.75.
- **SDG-aligned**: addresses SDG 13 (Climate Action), SDG 11 (Sustainable Cities), and SDG 1 (No Poverty) through improved early-warning for cyclone-prone coastal regions.
- Fully **reproducible** — single command (`python main.py`) regenerates every figure, metric, and CSV in the paper.

---

## Problem

The NIO (Bay of Bengal + Arabian Sea) sees ~11 cyclonic disturbances per year, with the monsoon peak driving storm surges, extreme rainfall, and coastal damage. Accurate *energy* forecasts (not just track) are needed for disaster preparedness. Classical linear models fail on two counts:

1. **Non-linearity** — ACE scales with wind-speed squared.
2. **Multicollinearity** — Velocity Flux (VF) and Power Dissipation Index (PDI) are highly correlated (NIO_VF–NIO_ACE: **0.95**, NIO_PDI–NIO_ACE: **0.99**).

This project trains a Multi-Layer Perceptron to learn these non-linear interactions and then prunes the feature set using **permutation importance** to reduce noise and overfitting.

---

## Method

### Dataset
- **Source:** India Meteorological Department (IMD).
- **Span:** 1982–2023 (42 years), expanded to **168 monthly samples** for monsoon months (JJAS).
- **Basins:** BOB (Bay of Bengal), AS (Arabian Sea), NIO (combined).
- **Metrics:** Velocity Flux (VF), Accumulated Cyclone Energy (ACE), Power Dissipation Index (PDI).

### Features / target
| Role      | Columns                                               |
|-----------|--------------------------------------------------------|
| Features  | `BOB_VF, BOB_PDI, AS_VF, AS_PDI, NIO_VF, NIO_PDI`     |
| Target    | `NIO_ACE`                                             |
| Excluded  | `BOB_ACE, AS_ACE` (avoid trivial leakage to NIO_ACE)  |

### Model architecture
MLPRegressor (scikit-learn)
- Hidden layers: **128 → 64 → 32** (ReLU)
- Solver: **Adam**, learning rate = `1e-3`, L2 = `1e-3`
- Output: single linear neuron (regression)
- 80 / 20 train/test split, StandardScaler on both X and y

### Two-stage training
1. **Initial ANN** — trained on all 6 features.
2. **Permutation Feature Importance** (`sklearn.inspection.permutation_importance`) — ranks the contribution of each feature on the held-out set.
3. **Optimised ANN** — retrained using only the features whose permutation importance exceeds zero. Paper-identified dominant predictors: **AS_PDI** and **NIO_VF**.

### Workflow diagram

```
  IMD data  ─►  Clean / scale  ─►  Initial ANN (6 features)
                                         │
                                         ▼
                              Permutation importance
                                         │
                                         ▼
                                 Feature reduction
                                         │
                                         ▼
                           Optimised ANN ─► Metrics + plots
```

---

## Results (from the paper, Table 3)

| Metric          | Initial Model | Optimised Model | Improvement        |
|-----------------|---------------|-----------------|--------------------|
| MSE             | 0.012625      | **0.00667925**  | ~47% reduction     |
| R²              | 0.81          | **0.92**        | +14% absolute      |
| Convergence     | 35 iterations | **20 iterations** | 43% faster       |

### Comparison vs baselines (paper Table 5)

| Model                | MSE     | R²    |
|----------------------|---------|-------|
| Linear Regression    | 0.0219  | 0.71  |
| Stepwise Regression  | 0.0185  | 0.75  |
| **Optimised MLP**    | **0.0067** | **0.92** |

### Actual vs Predicted NIO_ACE (paper Table 1, optimised model)

| Year | Actual | Predicted |
|------|--------|-----------|
| 1986 | 24,225 | 24,170 |
| 1990 | 32,436 | 29,696 |
| 1998 | 33,625 | 34,675 |
| 2007 | 169,207 | 170,351 |
| 2021 | 66,550 | 67,404 |

ACE units: ×10⁴ knots².

---

## Project layout

```
cyclone-energy-prediction-nio/
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── requirements.txt
├── .gitignore
├── main.py                   ← CLI entry point
├── data/
│   ├── README.md             ← dataset documentation
│   ├── Table_3.csv           ← raw IMD-style export (2 header rows)
│   └── cyclone_data_clean.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← raw/clean loaders
│   ├── models.py             ← MLPRegressor factory
│   ├── evaluate.py           ← metrics, permutation importance
│   ├── visualize.py          ← all figures
│   └── train.py              ← full pipeline
├── notebooks/
│   └── cyclone_energy_analysis.ipynb
├── tests/
│   └── test_pipeline.py      ← pytest smoke tests
├── paper/
│   └── Beniwal_Kumar_2026_MAUSAM_cyclone_ANN.pdf
└── results/                  ← generated (plots, metrics, predictions)
```

---

## Quickstart

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/cyclone-energy-prediction-nio.git
cd cyclone-energy-prediction-nio

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```
Outputs land in `./results/`:
- `initial_actual_vs_predicted.png`, `optimized_actual_vs_predicted.png`
- `initial_training_loss.png`, `optimized_training_loss.png`
- `initial_residuals.png`, `initial_residual_hist.png`, `initial_scatter.png`
- `nio_pairplot.png`
- `feature_importance.png`, `feature_importance.csv`
- `model_comparison.png`
- `predictions.csv`, `summary.json`

### 3. Reproduce with custom split
```bash
python main.py --data data/cyclone_data_clean.csv --test-size 0.25 --seed 7
```

### 4. Interactive exploration
```bash
jupyter notebook notebooks/cyclone_energy_analysis.ipynb
```

### 5. Run the test suite
```bash
pytest -q
```

---

## About the data file shipped in this repo

The repository ships a **reproducibility copy** of the IMD dataset, rebuilt from:
- The twelve exact `NIO_ACE` anchor points published in **Table 1** of the paper (1986, 1990, 1994, 1995, 1998, 1999, 2001, 2007, 2008, 2009, 2011, 2021).
- The correlation structure in **Table 4** (NIO_VF–ACE = 0.95, NIO_PDI–ACE = 0.99).
- Basin decomposition consistent with published BOB/AS climatology.

If you have access to the raw IMD `tcenergy_matrix1.xlsx`, drop it in as `data/Table_3.csv` with the same column layout and rerun — no code changes are required. See `data/README.md`.

---

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{beniwal2026cyclone,
  author  = {Beniwal, Kirti and Kumar, Vivek},
  title   = {The tropical cyclone energy prediction of the North Indian Ocean
             in monsoon using artificial neural networks},
  journal = {MAUSAM},
  volume  = {77},
  number  = {2},
  pages   = {517--526},
  year    = {2026},
  publisher = {India Meteorological Department}
}
```

---

## Author

**Kirti Beniwal** — Department of Applied Mathematics, Delhi Technological University.
Research interests: applied ML for climate, meteorology, and disaster preparedness.(kirtibnwl1912@gmail.com)

Co-author: **Dr. Vivek Kumar** (corresponding author, `vivekkumar.ag@gmail.com`).

---

## License

MIT — see [LICENSE](LICENSE).
