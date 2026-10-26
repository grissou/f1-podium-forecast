"""Build the base driver x race table that every feature and model starts from.

One row per driver per race. Columns are split into three groups:

- KEYS: identifiers and the time index used for splitting.
- PRE_RACE: facts known before lights out (starting grid, qualifying).
- OUTCOMES: what happened in the race. These are labels or inputs to
  *lagged* features only, never features for the same race.
"""

import pandas as pd

KEYS = ["raceId", "driverId", "constructorId", "year", "round", "date", "circuitId"]
PRE_RACE = ["grid", "pit_lane_start", "quali_position", "quali_best_s"]
OUTCOMES = ["position_order", "finished", "points", "status_id", "podium"]

PODIUM_POSITIONS = 3


def build_base_table(tables: dict[str, pd.DataFrame], min_year: int = 2010) -> pd.DataFrame:
    races = tables["races"][["raceId", "year", "round", "date", "circuitId"]]
    results = tables["results"]
    qualifying = tables["qualifying"][["raceId", "driverId", "position", "q1", "q2", "q3"]].rename(
        columns={"position": "quali_position"}
    )
    # Best lap across the three sessions: the cleanest measure of one-lap pace.
    laps = qualifying[["q1", "q2", "q3"]].apply(_lap_seconds)
    qualifying = qualifying.assign(quali_best_s=laps.min(axis=1)).drop(columns=["q1", "q2", "q3"])
    # A handful of races list a driver twice in qualifying; keep their best result.
    qualifying = qualifying.sort_values("quali_position").drop_duplicates(
        ["raceId", "driverId"], keep="first"
    )

    df = results[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder", "position",
         "points", "statusId"]
    ].merge(races, on="raceId", how="inner", validate="many_to_one")
    df = df[df["year"] >= min_year]
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left", validate="one_to_one")

    df = _fix_grid(df)

    df = df.rename(columns={"positionOrder": "position_order", "statusId": "status_id"})
    # `position` is null for drivers who were not classified.
    df["finished"] = df["position"].notna().astype(int)
    df["podium"] = (df["position_order"] <= PODIUM_POSITIONS).astype(int)

    df = df[KEYS + PRE_RACE + OUTCOMES]
    return df.sort_values(["date", "raceId", "grid"]).reset_index(drop=True)


def _lap_seconds(lap: pd.Series) -> pd.Series:
    """'1:26.572' -> 86.572; missing stays missing."""
    return pd.to_timedelta("0:" + lap, errors="coerce").dt.total_seconds()


def _fix_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Grid 0 means a pit-lane start: put those cars behind the last grid slot."""
    df = df.copy()
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    last_slot = df.loc[df["grid"] > 0].groupby("raceId")["grid"].max()
    back_of_grid = df["raceId"].map(last_slot).fillna(0) + 1
    df["grid"] = df["grid"].where(df["grid"] > 0, back_of_grid).astype(int)
    return df
