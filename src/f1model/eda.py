"""Phase 1 summary: dataset size and podium rate by grid slot.

Usage: python -m f1model.eda --data-dir data/raw
"""

import argparse
from pathlib import Path

from f1model.dataset import build_base_table
from f1model.load import load_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--min-year", type=int, default=2010)
    args = parser.parse_args()

    df = build_base_table(load_tables(args.data_dir), min_year=args.min_year)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Rows: {len(df):,}  races: {df['raceId'].nunique():,}  "
          f"seasons: {df['year'].min()}-{df['year'].max()}")
    print(f"Podium rate: {df['podium'].mean():.3f}")

    by_slot = (
        df.assign(slot=df["grid"].clip(upper=21))
        .groupby("slot")["podium"]
        .agg(podium_rate="mean", starts="count")
    )
    by_slot.to_csv(out / "podium_rate_by_grid.csv")
    print("\nPodium rate by grid slot (21 = 21st or further back):")
    print(by_slot.round(3).to_string())

    by_year = df.groupby("year").agg(races=("raceId", "nunique"), starters=("driverId", "count"))
    by_year.to_csv(out / "races_by_year.csv")


if __name__ == "__main__":
    main()
