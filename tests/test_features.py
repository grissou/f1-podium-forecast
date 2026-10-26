import pandas as pd
import pytest

from f1model.dataset import build_base_table
from f1model.features import FEATURES, build_features


def test_no_feature_looks_ahead(synthetic_base):
    """Changing race t's results must not change any feature for race t or earlier."""
    base = synthetic_base
    t = 10
    perturbed = base.copy()
    at_t = perturbed["raceId"] == t
    perturbed.loc[at_t, "position_order"] = perturbed.loc[at_t, "position_order"].to_numpy()[::-1]
    perturbed.loc[at_t, "podium"] = (perturbed.loc[at_t, "position_order"] <= 3).astype(int)
    perturbed.loc[at_t, "points"] = 0
    perturbed.loc[at_t, "finished"] = 0

    before = build_features(base)
    after = build_features(perturbed)
    date_t = base.loc[at_t, "date"].iloc[0]
    up_to_t = before["date"] <= date_t

    pd.testing.assert_frame_equal(before.loc[up_to_t, FEATURES], after.loc[up_to_t, FEATURES])
    # The test has teeth: later races do see the change.
    assert not before.loc[~up_to_t, FEATURES].equals(after.loc[~up_to_t, FEATURES])


def test_feature_values_on_fixture(tables):
    df = build_features(build_base_table(tables)).set_index(["raceId", "driverId"])

    # First race of the sample: no history, zero season points.
    assert df.loc[(2, 101), ["driver_podium_rate_5", "team_points_5", "driver_circuit_mean_finish"]].isna().all()
    assert df.loc[(2, 101), "driver_season_points"] == 0

    # Race 3 sees race 2: driver 101 was P2 (18 points, podium); team 2 scored 0 + 15.
    assert df.loc[(3, 101), "driver_podium_rate_5"] == 1
    assert df.loc[(3, 101), "driver_mean_finish_5"] == 2
    assert df.loc[(3, 101), "driver_season_points"] == 18
    assert df.loc[(3, 104), "team_podium_rate_5"] == 0.5
    assert df.loc[(3, 104), "team_points_5"] == 7.5
    assert df.loc[(3, 104), "team_season_points"] == 15
    assert df.loc[(3, 103), "driver_dnf_rate_10"] == 1

    # Qualifying gap: pole 89.0 s, driver 103 best 90.9 s.
    assert df.loc[(2, 101), "quali_gap_pct"] == 0
    assert df.loc[(2, 103), "quali_gap_pct"] == pytest.approx((90.9 / 89.0 - 1) * 100)


def test_features_keep_rows_and_order(tables):
    base = build_base_table(tables)
    df = build_features(base)
    assert len(df) == len(base)
    assert df["date"].is_monotonic_increasing
    assert set(FEATURES) <= set(df.columns)
