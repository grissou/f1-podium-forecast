import pandas as pd
import pytest


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
    qualifying = pd.DataFrame({
        "raceId":   [2, 2, 2, 2, 3, 3, 3],
        "driverId": [101, 102, 103, 104, 101, 102, 103],
        "position": [1, 2, 3, 4, 2, 1, 4],
    })
    return {"races": races, "results": results, "qualifying": qualifying}
