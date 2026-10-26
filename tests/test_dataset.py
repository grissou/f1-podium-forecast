import pandas as pd

from f1model.dataset import KEYS, OUTCOMES, PRE_RACE, build_base_table


def test_one_row_per_driver_per_race_from_min_year(tables):
    df = build_base_table(tables, min_year=2010)
    assert len(df) == 8
    assert df["year"].min() == 2010
    assert not df.duplicated(["raceId", "driverId"]).any()
    assert list(df.columns) == KEYS + PRE_RACE + OUTCOMES


def test_podium_label_and_finished_flag(tables):
    df = build_base_table(tables).set_index(["raceId", "driverId"])
    assert df.loc[(2, 102), "podium"] == 1
    assert df.loc[(2, 104), "podium"] == 1  # P3 from the pit lane
    assert df.loc[(2, 103), "podium"] == 0
    assert df.loc[(2, 103), "finished"] == 0
    assert df.groupby(level="raceId")["podium"].sum().eq(3).all()


def test_pit_lane_start_goes_behind_last_grid_slot(tables):
    df = build_base_table(tables).set_index(["raceId", "driverId"])
    assert df.loc[(2, 104), "grid"] == 4
    assert df.loc[(2, 104), "pit_lane_start"] == 1
    assert df["grid"].min() >= 1


def test_missing_qualifying_is_left_null(tables):
    df = build_base_table(tables).set_index(["raceId", "driverId"])
    assert df["quali_position"].isna().sum() == 1
    assert pd.isna(df.loc[(3, 104), "quali_position"])
    assert pd.isna(df.loc[(3, 104), "quali_best_s"])


def test_best_qualifying_lap_in_seconds(tables):
    df = build_base_table(tables).set_index(["raceId", "driverId"])
    assert df.loc[(2, 101), "quali_best_s"] == 89.0   # Q3 lap 1:29.000
    assert df.loc[(2, 104), "quali_best_s"] == 91.5   # only a Q1 lap


def test_rows_are_in_time_order(tables):
    df = build_base_table(tables)
    assert df["date"].is_monotonic_increasing
