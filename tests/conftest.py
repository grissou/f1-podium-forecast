import numpy as np
import pandas as pd
import pytest

from f1model.dataset import KEYS, OUTCOMES, PRE_RACE


@pytest.fixture
def tables():
    """Two tiny races in the raw Kaggle schema, plus one race before 2010."""
    races = pd.DataFrame({
        "raceId": [1, 2, 3],
        "year": [2009, 2010, 2010],
        "round": [1, 1, 2],
        "circuitId": [10, 11, 12],
        "date": pd.to_datetime(["2009-03-29", "2010-03-14", "2010-03-28"]),
    })
    # Race 2: four starters, driver 104 started from the pit lane (grid 0)
    # and driver 103 did not finish (position null).
    # Race 3: four starters, all classified.
    results = pd.DataFrame({
        "raceId":        [1, 2, 2, 2, 2, 3, 3, 3, 3],
        "driverId":      [101, 101, 102, 103, 104, 101, 102, 103, 104],
        "constructorId": [1, 1, 1, 2, 2, 1, 1, 2, 2],
        "grid":          [1, 1, 2, 3, 0, 2, 1, 4, 3],
        "positionOrder": [1, 2, 1, 4, 3, 1, 4, 2, 3],
        "position":      [1, 2, 1, None, 3, 1, 4, 2, 3],
        "points":        [10, 18, 25, 0, 15, 25, 12, 18, 15],
        "statusId":      [1, 1, 1, 5, 1, 1, 1, 1, 1],
    })
    # Driver 104 has no qualifying row for race 3.
    qualifying = pd.DataFrame({
        "raceId":   [2, 2, 2, 2, 3, 3, 3],
        "driverId": [101, 102, 103, 104, 101, 102, 103],
        "position": [1, 2, 3, 4, 2, 1, 4],
        "q1": ["1:30.000", "1:30.500", "1:31.000", "1:31.500", "1:20.500", "1:20.000", "1:21.000"],
        "q2": ["1:29.800", "1:30.200", "1:30.900", None, "1:20.400", "1:19.900", None],
        "q3": ["1:29.000", "1:29.500", None, None, None, None, None],
    })
    return {"races": races, "results": results, "qualifying": qualifying}


@pytest.fixture
def synthetic_base() -> pd.DataFrame:
    """Three seasons of six races on three circuits, eight drivers in four teams, random outcomes."""
    rng = np.random.default_rng(0)
    rows = []
    race_id = 0
    for year in (2010, 2011, 2012):
        for rnd in range(1, 7):
            race_id += 1
            order = rng.permutation(8) + 1
            grid = rng.permutation(8) + 1
            finished = rng.random(8) > 0.15
            for d in range(8):
                rows.append({
                    "raceId": race_id, "driverId": 100 + d, "constructorId": d // 2,
                    "year": year, "round": rnd, "circuitId": rnd % 3,
                    "date": pd.Timestamp(f"{year}-03-01") + pd.Timedelta(days=14 * rnd),
                    "grid": grid[d], "pit_lane_start": 0, "quali_position": grid[d],
                    "quali_best_s": 80 + rng.random(),
                    "position_order": order[d], "finished": int(finished[d]),
                    "points": max(0, 11 - order[d]) if finished[d] else 0,
                    "status_id": 1 if finished[d] else 5, "podium": int(order[d] <= 3),
                })
    return pd.DataFrame(rows)[KEYS + PRE_RACE + OUTCOMES]
