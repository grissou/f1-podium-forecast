"""Expanding-window walk-forward splits by season.

Train on every season before the test season, score that season, roll
forward one season. Labels resolve on race day and all features are lagged,
so no purge gap is needed between train and test.
"""

import pandas as pd

TEST_SEASONS = range(2016, 2023)
HOLDOUT_SEASONS = (2023, 2024)


def walk_forward(df: pd.DataFrame, test_seasons=TEST_SEASONS):
    """Yield (test_year, train_index, test_index) for each test season."""
    for year in test_seasons:
        yield year, df.index[df["year"] < year], df.index[df["year"] == year]
