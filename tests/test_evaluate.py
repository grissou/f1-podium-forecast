import numpy as np
import pandas as pd

from f1model.baseline import GridBaseline
from f1model.evaluate import (METRICS, compare, expected_calibration_error, holdout,
                              reliability_table, top3_hit_rate)
from f1model.features import build_features
from f1model.train import logistic_regression


def test_top3_hit_rate_picks_highest_probabilities():
    df = pd.DataFrame({"raceId": [1] * 4 + [2] * 4, "podium": [1, 1, 1, 0, 1, 1, 1, 0]})
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.1, 0.8, 0.7])  # race 2: one pick misses
    assert top3_hit_rate(df, p) == 5 / 6


def test_ece_is_zero_when_calibrated():
    y = np.array([1, 0, 0, 0] * 50)   # 25% positives
    p = np.full(200, 0.25)
    assert expected_calibration_error(y, p) == 0


def test_backtest_and_brier_skill(synthetic_base):
    df = build_features(synthetic_base)
    results = compare({"grid": GridBaseline, "logistic": logistic_regression}, df, test_seasons=[2011, 2012])
    assert list(results.index.names) == ["model", "season"]
    assert len(results) == 4
    assert results[METRICS].notna().all().all()
    assert (results.loc["grid", "brier_skill"] == 0).all()


def test_holdout_fits_once_before_the_holdout_seasons(synthetic_base):
    df = build_features(synthetic_base)
    results = holdout({"grid": GridBaseline}, df, seasons=(2012,))
    assert list(results.index) == [("grid", "2012"), ("grid", "all")]
    assert results[METRICS].notna().all().all()


def test_reliability_table_bins_predictions():
    table = reliability_table(y=[0, 0, 1, 1], p=[0.1, 0.2, 0.8, 0.9], edges=[0, 0.5, 1.0])
    assert list(table["n"]) == [2, 2]
    assert list(table["observed"]) == [0.0, 1.0]
