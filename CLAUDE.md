# CLAUDE.md

Pre-race F1 podium probability model (XGBoost) on the Kaggle dataset
`rohanrao/formula-1-world-championship-1950-2020`. It is a portfolio project
for finance / quant applications, so methodological rigour matters more than
squeezing out accuracy.

Plan: https://claude.ai/artifact/NXjraZiC8UwTbjfVhBEVgi

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                       # unit tests, no data needed
python -m f1model.eda --data-dir data/raw    # phase 1 summary (needs data)
```

Data: CSVs go in `data/raw/` (gitignored, never commit them). Download with
`kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")`
and copy the `*.csv` files over. Missing values in the CSVs are the literal `\N`.

## Design rules

- **Unit of prediction:** one row per driver per race, 2010 onward. Target
  `podium` = finishing order <= 3.
- **No look-ahead.** A feature for race *t* may use only races before *t*,
  plus pre-race facts for *t* (grid, qualifying). Columns in
  `dataset.OUTCOMES` are labels or inputs to lagged features only.
- `driver_standings` / `constructor_standings` are after the race: always join
  the previous round's standings.
- Rolling features: `groupby(entity).shift(1)` before any rolling or expanding
  window. Never use full-season aggregates.
- **Validation:** expanding-window walk-forward by season (test 2016 to 2022),
  tuning on an inner walk-forward inside each training window. Never random
  K-fold. 2023 and 2024 are a holdout, run once at the end.
- Fit encoders, imputers and calibrators inside each training window only.
- Benchmark: `GridBaseline` (podium rate by grid slot). Report log loss,
  Brier skill vs the baseline, ROC/PR-AUC and calibration per season.
- Every new feature needs a leakage test: its value for race *t* must not
  change when race *t*'s results are removed.

## Code layout

`src/f1model/`: `load.py` (CSV reading), `dataset.py` (base table),
`baseline.py` (grid benchmark), `eda.py` (phase 1 script). Planned:
`features.py`, `splits.py`, `train.py`, `evaluate.py`, `explain.py`.
Tests use the small synthetic fixture in `tests/conftest.py`; keep CI
independent of the real data.
