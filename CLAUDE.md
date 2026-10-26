# CLAUDE.md

Pre-race F1 podium probability model (XGBoost) on the Kaggle dataset
`rohanrao/formula-1-world-championship-1950-2020`. It is a portfolio project
for finance / quant applications, so methodological rigour matters more than
squeezing out accuracy.

Plan: https://claude.ai/artifact/NXjraZiC8UwTbjfVhBEVgi

## Priority 1: clarity, small size, explainability

These beat accuracy and features whenever they conflict.

- **Clarity:** code a reader can follow top to bottom. Plain pandas, short
  functions, descriptive names, a one-line docstring saying *why*.
- **Minimal size:** fewest files, lines and dependencies that do the job. No
  speculative abstractions, config systems or helpers used once. Delete before
  adding. Prefer a few strong features over many weak ones.
- **Explainability:** every feature has a plain-English meaning and a finance
  analogue; every modelling choice is justified in one sentence; SHAP output
  must be readable by a non-specialist. If a gain in score cannot be
  explained, it does not go in.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                       # unit tests, no data needed
python -m f1model.eda --data-dir data/raw    # phase 1 summary (needs data)
python -m f1model.evaluate --data-dir data/raw  # walk-forward backtest + reliability table
python -m f1model.evaluate --data-dir data/raw --holdout  # + 2023-2024 holdout (already run once)
python -m f1model.explain --data-dir data/raw   # SHAP values + README figures
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
- **Sprints are excluded.** `sprint_results` is not loaded and season points use
  race points only, so 2021+ ranks differ slightly from official standings.
- Rolling features: `groupby(entity).shift(1)` before any rolling or expanding
  window. Never use full-season aggregates.
- **Validation:** expanding-window walk-forward by season (test 2016 to 2022),
  tuning on an inner walk-forward inside each training window. Never random
  K-fold. 2023 and 2024 are a holdout, run once at the end.
- Fit encoders, imputers and calibrators inside each training window only.
- **XGBoost stays on `XGB_DEFAULTS`.** The last training season picks the
  number of rounds by early stopping. A 20-draw random search on that season
  lowered out-of-sample Brier skill (6.2% to 4.9%), so tuning is off by default.
- **No calibrator.** Raw XGBoost probabilities are calibrated (mean predicted
  = observed podium rate); Platt and isotonic fitted on the inner season did
  not help. Reliability table is in the README.
- **The holdout has been run.** 2023-2024 results are in the README. Do not
  change the model and re-score the holdout; that would make it a test set.
- **One README.** `README.md` is the single write-up and must explain every
  design choice and carry every reported number. No other READMEs.
- Benchmark: `GridBaseline` (podium rate by grid slot). Report log loss,
  Brier skill vs the baseline, ROC/PR-AUC and calibration per season. Numbers
  in the README come from `python -m f1model.evaluate`; rerun it after any
  change and update the README tables.
- Every new feature needs a leakage test: its value for race *t* must not
  change when race *t*'s results are removed.

## Code layout

`src/f1model/`: `load.py` (CSV reading), `dataset.py` (base table),
`features.py` (point-in-time features), `baseline.py` (grid benchmark),
`splits.py` (walk-forward folds), `train.py` (models with a shared
`fit(df)` / `predict_proba(df)` interface), `evaluate.py` (metrics, backtest
script), `explain.py` (out-of-sample SHAP and all figures), `eda.py` (phase 1
script). All six phases are done; figures live in `reports/figures/` and are
committed.
Tests use the small synthetic fixture in `tests/conftest.py`; keep CI
independent of the real data.
