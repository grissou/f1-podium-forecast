# f1-podium-forecast

Pre-race podium probability model for Formula 1, built with XGBoost and
validated with a season-by-season walk-forward backtest (no look-ahead bias).

For every driver in every race since 2010, the model predicts P(top-3 finish)
using only information available before the start. Its benchmark is the
starting grid: the model is only useful if it beats a forecast built from grid
position alone.

## Status

- [x] Phase 1: data loading, clean driver × race table, grid-position baseline
- [ ] Phase 2: point-in-time features and leakage tests
- [ ] Phase 3: walk-forward harness and baselines
- [ ] Phase 4: XGBoost and tuning
- [ ] Phase 5: calibration and 2023–2024 holdout
- [ ] Phase 6: SHAP and write-up

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Download the data as described in [data/README.md](data/README.md), then:

```bash
python -m f1model.eda --data-dir data/raw
```

This prints the dataset size and the podium rate by grid slot, and writes
both to `reports/`.

## Layout

```
src/f1model/
  load.py       read and type the Kaggle CSVs
  dataset.py    one row per driver per race; pre-race columns vs outcomes
  baseline.py   benchmark: podium rate by grid slot
  eda.py        phase 1 summary script
tests/          unit tests on a synthetic two-race fixture
```
