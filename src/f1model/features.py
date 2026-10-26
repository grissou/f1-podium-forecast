"""Point-in-time features built from the base table.

A feature for race t may use only races before t (every rolling window is
shifted by one race first) plus pre-race facts for t (grid, qualifying).
Nothing here reads a column in dataset.OUTCOMES without that shift.

Finance analogues: form = momentum, team form = sector momentum, season
points = relative strength, DNF rate = default risk, circuit history =
seasonality.
"""

import pandas as pd

FEATURES = [
    # pre-race facts for this race
    "grid", "pit_lane_start", "quali_position", "quali_gap_pct",
    # driver form over previous races
    "driver_podium_rate_5", "driver_podium_rate_10", "driver_mean_finish_5", "driver_dnf_rate_10",
    # team form over previous races, both cars
    "team_podium_rate_5", "team_points_5",
    # championship state before this race
    "driver_season_points", "driver_season_rank", "team_season_points",
    # this driver's past results at this circuit
    "driver_circuit_mean_finish",
]


def build_features(base: pd.DataFrame) -> pd.DataFrame:
    """Return the base table with FEATURES added, one row per driver per race in time order."""
    df = base.sort_values(["date", "raceId", "grid"]).reset_index(drop=True)
    df = _qualifying(df)
    df = _driver_form(df)
    df = _team_form(df)
    df = _season_points(df)
    df = _circuit_history(df)
    return df


def _lagged_mean(values: pd.Series, by: pd.Series, window: int | None) -> pd.Series:
    """Mean of `values` over the previous `window` rows in each group, excluding the current row.

    `window=None` means every previous row (an expanding window).
    """
    def one_group(s: pd.Series) -> pd.Series:
        past = s.shift(1)
        return past.expanding().mean() if window is None else past.rolling(window, min_periods=1).mean()

    return values.groupby(by).transform(one_group)


def _qualifying(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to the fastest qualifier in percent: 0 for pole, ~1 for a midfield car."""
    pole = df.groupby("raceId")["quali_best_s"].transform("min")
    df["quali_gap_pct"] = (df["quali_best_s"] / pole - 1) * 100
    return df


def _driver_form(df: pd.DataFrame) -> pd.DataFrame:
    driver = df["driverId"]
    df["driver_podium_rate_5"] = _lagged_mean(df["podium"], driver, 5)
    df["driver_podium_rate_10"] = _lagged_mean(df["podium"], driver, 10)
    df["driver_mean_finish_5"] = _lagged_mean(df["position_order"], driver, 5)
    df["driver_dnf_rate_10"] = _lagged_mean(1 - df["finished"], driver, 10)
    return df


def _team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Average the two cars per race first, then roll over the team's previous races."""
    team = (
        df.groupby(["constructorId", "raceId", "date"], as_index=False)
        .agg(podium=("podium", "mean"), points=("points", "mean"))
        .sort_values(["date", "raceId"])
    )
    team["team_podium_rate_5"] = _lagged_mean(team["podium"], team["constructorId"], 5)
    team["team_points_5"] = _lagged_mean(team["points"], team["constructorId"], 5)
    cols = ["constructorId", "raceId", "team_podium_rate_5", "team_points_5"]
    return df.merge(team[cols], on=["constructorId", "raceId"], how="left")


def _season_points(df: pd.DataFrame) -> pd.DataFrame:
    """Championship points scored earlier in the same season, and the resulting rank.

    Computed from race results rather than the standings tables, which are
    recorded after each race. Sprint points (2021 on) are not included.
    """
    season_driver = df.groupby(["year", "driverId"])["points"]
    df["driver_season_points"] = season_driver.cumsum() - df["points"]
    df["driver_season_rank"] = (
        df.groupby("raceId")["driver_season_points"].rank(ascending=False, method="min")
    )
    team = (
        df.groupby(["year", "constructorId", "raceId", "date"], as_index=False)["points"].sum()
        .sort_values(["date", "raceId"])
    )
    team["team_season_points"] = team.groupby(["year", "constructorId"])["points"].cumsum() - team["points"]
    return df.merge(team[["constructorId", "raceId", "team_season_points"]],
                    on=["constructorId", "raceId"], how="left")


def _circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """Mean finishing order in this driver's previous visits to this circuit."""
    df["driver_circuit_mean_finish"] = _lagged_mean(
        df["position_order"], [df["driverId"], df["circuitId"]], None
    )
    return df
