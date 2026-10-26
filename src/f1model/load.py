"""Read the Kaggle F1 CSVs into typed DataFrames."""

from pathlib import Path

import pandas as pd

# The dataset writes missing values as the literal string "\N".
NA_VALUES = ["\\N"]

TABLES = (
    "races",
    "results",
    "qualifying",
    "drivers",
    "constructors",
    "circuits",
    "status",
    "driver_standings",
    "constructor_standings",
)


def load_table(data_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README.md for how to download the dataset."
        )
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_tables(data_dir: str | Path, names=TABLES) -> dict[str, pd.DataFrame]:
    return {name: load_table(data_dir, name) for name in names}
