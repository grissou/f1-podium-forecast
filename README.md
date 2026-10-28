# F1 podium forecast

**Before a Formula 1 race starts, which drivers will finish on the podium, and
can a model say so better than the starting grid alone?**

A pre-race podium probability model built with XGBoost on 15 seasons of
Formula 1 (2010 to 2024), validated season by season with no look-ahead,
judged against the starting grid as the benchmark, and explained with SHAP.

Over seven out-of-sample test seasons (2016 to 2022) the model beats the grid
in every season (2019 by a margin within run-to-run noise), cutting log loss
from 0.231 to 0.203 and adding 6.2% Brier skill. On the untouched 2023 to 2024
holdout it wins in 2023 and loses in 2024, netting a small log-loss gain and no
Brier skill. Hyperparameter tuning was tried and made things worse out of
sample, so the model uses fixed defaults.

| | |
|---|---|
| Rows | 6,436 driver-race pairs, 305 races, 2010 to 2024 |
| Target | finished 1st, 2nd or 3rd (14.2% of rows) |
| Features | 14, all computable before the race |
| Validation | expanding-window walk-forward, test 2016 to 2022; holdout 2023 to 2024 |
| Model | XGBoost, depth 3, fixed defaults, early-stopped on the last training season |
| Code | 8 modules, about 670 lines, 21 unit tests, CI |

## The question

The obvious forecast is the starting grid: since 2010, pole position has
finished on the podium 80% of the time, second on the grid 69%, third 51%, and
from eighth back under 5%. So every model here is compared with a benchmark
that knows only the grid slot and its historical podium rate. The model has to
add something the grid does not already contain.

## Results

### Walk-forward, test seasons 2016 to 2022

Each test season is scored by a model trained on every season before it.
Means over the seven seasons; lower is better for log loss and Brier.

| Model | Log loss | Brier | ROC-AUC | PR-AUC | Calibration error | Top-3 hit rate | Brier skill vs grid |
|---|---|---|---|---|---|---|---|
| Grid baseline | 0.231 | 0.068 | 0.922 | 0.670 | 0.025 | 69.9% | 0 |
| Logistic regression | 0.212 | 0.065 | 0.938 | 0.713 | 0.025 | 68.8% | +4.2% |
| **XGBoost, fixed defaults** | **0.203** | **0.064** | **0.944** | **0.729** | 0.031 | 69.9% | **+6.2%** |
| XGBoost, tuned per window | 0.205 | 0.065 | 0.943 | 0.717 | 0.033 | 71.1% | +4.9% |

![Out-of-sample log loss by test season](reports/figures/backtest_log_loss.png)

Brier skill against the grid, season by season:

| Test season | Logistic | XGBoost | XGBoost tuned |
|---|---|---|---|
| 2016 | +4.6% | +6.0% | +3.3% |
| 2017 | +2.3% | +5.3% | +5.8% |
| 2018 | +4.3% | +9.6% | +9.5% |
| 2019 | -4.6% | +1.2% | -4.3% |
| 2020 | +7.6% | +7.4% | +6.3% |
| 2021 | +5.2% | +9.6% | +9.4% |
| 2022 | +10.2% | +4.4% | +4.4% |

2019 was hard for every model: Mercedes started at the front and finished
there, so the grid was almost perfectly informative. The 2019 margin is thin
enough to depend on the computer: the same code and library versions on an
x86 Linux machine give -1.3% for 2019 and +5.7% on average, because XGBoost's
row sampling plays out slightly differently. The numbers here come from an
Apple-silicon Mac with the versions pinned in `pyproject.toml`.

Accuracy is not reported: predicting "no podium" for everyone is already 86%
accurate. Log loss is the primary metric; Brier skill is 1 minus model Brier
over grid Brier. The top-3 hit rate barely moves between models because all
of them pick the front row; the difference is in how they price the midfield.

### Holdout, 2023 to 2024

Never used for any decision. Every model was trained once on 2010 to 2022 and
scored once.

| Model | Season | Log loss | Brier | ROC-AUC | Brier skill vs grid |
|---|---|---|---|---|---|
| Grid baseline | 2023 | 0.257 | 0.074 | 0.896 | 0 |
| | 2024 | 0.230 | 0.069 | 0.920 | 0 |
| | both | 0.243 | 0.071 | 0.909 | 0 |
| Logistic regression | 2023 | 0.224 | 0.067 | 0.927 | +10.1% |
| | 2024 | 0.233 | 0.072 | 0.927 | -4.5% |
| | both | 0.228 | 0.069 | 0.928 | +2.8% |
| **XGBoost, fixed defaults** | 2023 | 0.226 | 0.069 | 0.928 | +7.5% |
| | 2024 | 0.237 | 0.073 | 0.926 | -7.2% |
| | both | 0.231 | 0.071 | 0.928 | +0.1% |
| XGBoost, tuned per window | 2023 | 0.222 | 0.068 | 0.929 | +8.4% |
| | 2024 | 0.223 | 0.069 | 0.933 | -1.2% |
| | both | 0.223 | 0.069 | 0.932 | +3.6% |

In 2023 every model beat the grid comfortably. In 2024 none did on Brier
score. Pooled, XGBoost improves log loss (0.243 to 0.231) and ROC-AUC (0.909
to 0.928) but its Brier skill is flat. The tuned model did better on the
holdout, but it was worse on the seven development seasons, and picking
whichever variant wins on the holdout would turn it into another training set.

### Is the edge real?

The model's log-loss gain over the grid, averaged per race, with a 95%
confidence interval from resampling races (not drivers: exactly three drivers
in each race podium, so they are not independent).

| Period | Gain per race | 95% CI | Races where the model was better |
|---|---|---|---|
| Walk-forward 2016 to 2022 | +0.028 | +0.019 to +0.038 | 72% of 144 |
| Holdout 2023 | +0.031 | +0.002 to +0.060 | 64% of 22 |
| Holdout 2024 | -0.007 | -0.022 to +0.011 | 25% of 24 |
| Holdout pooled | +0.011 | -0.006 to +0.029 | 43% of 46 |

Over the walk-forward seasons the interval is well clear of zero. On the
holdout the edge held in 2023 and vanished in 2024; pooled over 46 races it
is positive but not distinguishable from zero. Nine out-of-sample seasons:
seven won clearly, 2019 a coin flip, 2024 lost. A season where the grid is
unusually informative (2024's grid log loss was among the lowest of any year,
as in 2019) is enough to erase a small edge. In the holdout the model was
better in fewer than half the races yet had a positive mean gain: its wins
are large and rare (a strong car starting far back), its losses small and
frequent.

## What drives the forecasts (SHAP)

SHAP values are computed on each walk-forward test season from the model
fitted on the seasons before it, so the explanations are out of sample too.

![Mean absolute SHAP value per feature](reports/figures/shap_importance.png)

Three features carry the model in roughly equal measure: grid slot (0.66),
the gap to the pole lap in percent (0.63) and the team's recent points (0.62),
each moving a typical forecast by about two-thirds of a log-odds unit. Then
come qualifying position (0.41) and the driver's podium rate over the last ten
races (0.40).

- **The qualifying gap adds information beyond grid order.** Pole by half a
  second and pole by a hundredth are the same grid slot but different
  forecasts.
- **The car matters as much as the start.** A strong car that qualified badly
  (a penalty, a wet session, a mistake) still has a real podium chance, and
  this is where most of the edge over the grid comes from.

![Feature influence by test season](reports/figures/shap_by_season.png)

Grid position became more important from 2020 on while the qualifying gap
became less so, consistent with closer qualifying margins. Championship rank
mattered most in 2016 and little afterwards.

![One forecast taken apart](reports/figures/shap_waterfall.png)

Lewis Hamilton, 2016 Belgian Grand Prix, starting 21st after engine penalties.
The base rate is a 13% podium chance (log-odds -1.89). Recent form (+0.79,
+0.65), team points (+0.48) and championship position (+0.26) pull it up; grid
slot (-0.63), qualifying gap (-0.53) and qualifying position (-0.43) pull it
down. Net forecast: 19%, against about 1% from the grid baseline. He finished
third.

## Data

The Kaggle dataset
[rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
(despite the name it runs to the end of 2024). Four tables are used:
`results`, `races`, `qualifying` and `drivers`. The CSVs are not committed
(`data/` is git-ignored); "Reproduce it" below shows how to download them.

- **2010 onwards only**: the start of the current 25-point system, one regime
  for points, grid sizes and data coverage.
- **Sprints are excluded.** Season-points features use Grand Prix points only,
  so ranks for 2021 to 2024 differ slightly from the official standings.
- **Pit-lane starts** are coded as grid 0 in the data. They are moved to one
  slot behind the last car, with a flag, rather than read as better than pole.
- **Standings tables are not used**: they are recorded after each race, so
  championship points are recomputed from results instead.

## Features

Fourteen features, built in `features.py`. For race *t*, each uses only races
before *t* plus pre-race facts for *t*; every rolling window is shifted back
one race before it is computed.

| Family | Feature | Meaning |
|---|---|---|
| Starting position | `grid` | Where the car starts |
| | `pit_lane_start` | Starting from the pit lane (1/0) |
| | `quali_position` | Qualifying classification |
| | `quali_gap_pct` | How far off the fastest qualifying lap, in percent (0 for pole) |
| Driver form | `driver_podium_rate_5`, `_10` | Share of the driver's last 5 / 10 races ending on the podium |
| | `driver_mean_finish_5` | Average finishing order over the last 5 races |
| | `driver_dnf_rate_10` | Share of the last 10 races not classified |
| Team form | `team_podium_rate_5` | Podium rate of the team's two cars over its last 5 races |
| | `team_points_5` | Mean points per car per race over the team's last 5 races |
| Championship | `driver_season_points` | Points scored earlier this season |
| | `driver_season_rank` | Rank of that total among drivers in this race |
| | `team_season_points` | Team's points earlier this season |
| Track history | `driver_circuit_mean_finish` | Average finish in previous visits to this circuit since 2010 |

Missing values (a driver's first race, a first visit to a circuit) are left
missing; XGBoost handles them natively and the logistic model imputes the
median within its training window.

## Leakage

| Trap | How it is avoided |
|---|---|
| Standings rows for race *t* already include race *t*'s points | Season points are cumulated from results, current race excluded |
| Outcome columns (`position`, `points`, `statusId`...) describe the race being predicted | Kept in a separate `OUTCOMES` group, never in the feature list |
| A rolling mean that includes the current race | Every window is `shift(1)` then roll |
| Random K-fold mixes future races into training | Splits are strictly by season |
| Preprocessing fitted on all data | Imputer, scaler and model are fitted inside each training window |

`tests/test_features.py` builds features on a synthetic three-season table,
scrambles the results of one mid-sample race and rebuilds. Every feature for
that race and earlier must be bit-for-bit unchanged, and at least one later
race must change, so the test cannot pass trivially. It runs in CI with no
real data.

## Validation and models

```
train 2010-2015  ->  test 2016
train 2010-2016  ->  test 2017
   ...
train 2010-2021  ->  test 2022
train 2010-2022  ->  HOLDOUT 2023-2024, scored once
```

Inside each training window the last season is held out again to pick the
number of boosting rounds by early stopping (and the hyperparameters, when
tuning is on); the model is then refitted on the whole window. Every label
resolves on race day and every feature is lagged, so no purge gap is needed.

- **Grid baseline** (`baseline.py`): the training-window podium rate for each
  grid slot, lightly smoothed toward the overall rate; slots beyond 20th are
  pooled.
- **Logistic regression** (`train.py`): median imputation, scaling, L2. It
  gets +4.2% of XGBoost's +6.2%, so most of the gain is the features, not the
  non-linearity.
- **XGBoost** (`train.py`): depth 3, learning rate 0.05, min child weight 5,
  row and column subsampling 0.8, early stopping with patience 50 (typically
  90 to 210 rounds).

A 20-draw random search on the inner season improved inner log loss by under
0.005 per fold, picked different settings every season, and was worse out of
sample (+4.9% vs +6.2% Brier skill). One season of about 400 rows is too small
to tune on. The search code remains so the result can be reproduced.

## Calibration

![Reliability diagram](reports/figures/reliability.png)

Pooled over 2016 to 2022, the model predicts a 14.8% podium rate and 14.8% is
observed. Platt scaling fitted on the inner season changed nothing meaningful
(log loss 0.204); isotonic regression overfitted (0.233). No calibrator is
used.

Each driver is scored on their own, so one race's probabilities are not
forced to sum to three. They average 3.01, with 90% of races between 2.5 and 3.5.
Rescaling each race to sum to three moves log loss from
0.2024 to 0.2013, so the raw probabilities are reported. A conditional-logit
or Plackett-Luce model would build the constraint in.

## Limitations

- The grid is set the day before on the same track in the same car, so it is
  close to the best possible predictor. A gain of 0.03 in log loss against it
  is an edge, not a large one, and the holdout is mixed.
- No pace features yet: `lap_times` and `pit_stops`, weather and regulation
  era are the obvious next inputs, each through the same leakage test.
- The dataset has no odds, so nothing here says the model would make money.
  The claims are about forecast skill against the grid.

## Reproduce it

Python 3.12 or newer. Library versions are pinned so the numbers reproduce.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 21 unit tests, no data needed
```

Download the data (no Kaggle account needed):

```bash
pip install kagglehub
python -c "
import kagglehub, shutil, pathlib
src = pathlib.Path(kagglehub.dataset_download('rohanrao/formula-1-world-championship-1950-2020'))
dst = pathlib.Path('data/raw'); dst.mkdir(parents=True, exist_ok=True)
for f in src.rglob('*.csv'): shutil.copy(f, dst / f.name)
"
```

Then:

```bash
python -m f1model.evaluate --data-dir data/raw --holdout  # walk-forward, holdout, bootstrap; ~40 s
python -m f1model.explain --data-dir data/raw             # SHAP values and figures; ~15 s
```

Every number and figure in this README comes from those commands. Tables land
in `reports/` as CSV and figures in `reports/figures/`.

## Layout

```
src/f1model/
  load.py        read the Kaggle CSVs
  dataset.py     one row per driver per race; keys, pre-race facts, outcomes
  features.py    the 14 point-in-time features
  baseline.py    grid baseline
  splits.py      walk-forward folds and holdout
  train.py       logistic and XGBoost
  evaluate.py    metrics, walk-forward, holdout, bootstrap
  explain.py     out-of-sample SHAP and the figures
tests/           21 tests, including the leakage test
```

The code was written with AI coding assistance (Claude Code) under my
direction.

## Licence

MIT.
