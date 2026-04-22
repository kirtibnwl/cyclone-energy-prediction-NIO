# Dataset

## Files
- **`Table_3.csv`** — Raw-style export matching the original IMD spreadsheet layout (two header rows before the data). The pipeline reads this through `src.data_loader.load_raw`, which drops the header rows and normalises column names.
- **`cyclone_data_clean.csv`** — Already-cleaned version with one header row. This is what `main.py` loads by default.

## Source
The original cyclone metrics (1982–2023) were provided by the **India Meteorological Department (IMD)** for the study by Beniwal & Kumar (2026). Raw IMD records covering the Bay of Bengal (BOB), Arabian Sea (AS), and the combined North Indian Ocean (NIO) basins.

> The dataset shipped with this repository is a **reproducibility copy** derived from the anchor values published in Table 1 of the paper (exact NIO_ACE for twelve years between 1986–2021), interpolated and perturbed to recreate the 1982–2023 monsoon series while preserving the correlations reported in Table 4 (NIO_VF ↔ NIO_ACE ≈ 0.95, NIO_PDI ↔ NIO_ACE ≈ 0.99).

If you have access to the full IMD export (e.g. the `tcenergy_matrix1.xlsx` referenced in the author's working folder), drop it in as `Table_3.csv` with the same column layout — no code changes required.

## Column reference
| Column  | Meaning                                              | Units              |
|---------|------------------------------------------------------|--------------------|
| Year    | Calendar year                                        | —                  |
| BOB_VF  | Bay of Bengal — Velocity Flux                        | knots              |
| BOB_ACE | Bay of Bengal — Accumulated Cyclone Energy           | 10⁴ knots²         |
| BOB_PDI | Bay of Bengal — Power Dissipation Index              | 10⁴ knots³         |
| AS_VF   | Arabian Sea — Velocity Flux                          | knots              |
| AS_ACE  | Arabian Sea — Accumulated Cyclone Energy             | 10⁴ knots²         |
| AS_PDI  | Arabian Sea — Power Dissipation Index                | 10⁴ knots³         |
| NIO_VF  | North Indian Ocean (combined) — Velocity Flux        | knots              |
| NIO_ACE | North Indian Ocean — Accumulated Cyclone Energy      | 10⁴ knots²         |
| NIO_PDI | North Indian Ocean — Power Dissipation Index         | 10⁴ knots³         |

## Target / features
- **Target:** `NIO_ACE`
- **Features:** `BOB_VF, BOB_PDI, AS_VF, AS_PDI, NIO_VF, NIO_PDI`
- `BOB_ACE` and `AS_ACE` are deliberately excluded from the feature set to avoid trivial leakage into `NIO_ACE`.
