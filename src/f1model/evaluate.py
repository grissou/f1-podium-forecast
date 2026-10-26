"""Score probability forecasts season by season and against the grid baseline.

Usage:
    python -m f1model.evaluate --data-dir data/raw             # walk-forward backtest
    python -m f1model.evaluate --data-dir data/raw --holdout   # plus the one-shot 2023-2024 holdout
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from f1model.baseline import GridBaseline
from f1model.dataset import build_base_table
from f1model.explain import plot_backtest, plot_reliability
from f1model.features import build_features
from f1model.load import load_tables
from f1model.splits import HOLDOUT_SEASONS, TEST_SEASONS, walk_forward
from f1model.train import logistic_regression, xgboost, xgboost_tuned

METRICS = ["log_loss", "brier", "roc_auc", "pr_auc", "ece", "top3_hit_rate"]
MODELS = {"grid": GridBaseline, "logistic": logistic_regression,
          "xgb": xgboost, "xgb_tuned": xgboost_tuned}
RELIABILITY_BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def score(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["podium"]
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ece": expected_calibration_error(y, p),
        "top3_hit_rate": top3_hit_rate(df, p),
    }


def reliability_table(y, p, edges) -> pd.DataFrame:
    """Per probability bin: how many forecasts, what they predicted on average, what happened."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.cut(frame["p"], edges, include_lowest=True)
    return frame.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Does 30% mean 30%? Mean gap between predicted and observed rate, weighted by bin size."""
    table = reliability_table(y, p, np.linspace(0, 1, bins + 1))
    return float(((table["predicted"] - table["observed"]).abs() * table["n"] / table["n"].sum()).sum())


def top3_hit_rate(df: pd.DataFrame, p: np.ndarray) -> float:
    """Of the three highest-probability drivers per race, the share that finished on the podium."""
    picks = df.assign(p=p).sort_values("p", ascending=False).groupby("raceId").head(3)
    return float(picks["podium"].mean())


def log_loss_gain(df: pd.DataFrame, p_model, p_grid, n_boot: int = 5000, seed: int = 0) -> dict:
    """Is the edge real? Per-race log-loss gain of the model over the grid, with a bootstrap CI.

    Races are the unit of resampling because drivers in one race are not
    independent (exactly three of them finish on the podium).
    """
    y = df["podium"].to_numpy()
    row_gain = _row_log_loss(y, np.asarray(p_grid)) - _row_log_loss(y, np.asarray(p_model))
    race_gain = pd.Series(row_gain, index=df.index).groupby(df["raceId"]).mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(race_gain, len(race_gain)).mean() for _ in range(n_boot)])
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"gain": race_gain.mean(), "ci_low": low, "ci_high": high,
            "share_better": (race_gain > 0).mean(), "races": len(race_gain)}


def _row_log_loss(y, p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward_predictions(make_model, df: pd.DataFrame, test_seasons=TEST_SEASONS) -> pd.Series:
    """Out-of-sample P(podium) for every row in the test seasons, from a fresh fit per season."""
    p = pd.Series(np.nan, index=df.index)
    for _, train, test in walk_forward(df, test_seasons):
        p[test] = make_model().fit(df.loc[train]).predict_proba(df.loc[test])
    return p.dropna()


def backtest(make_model, df: pd.DataFrame, test_seasons=TEST_SEASONS) -> pd.DataFrame:
    """One row of METRICS per test season."""
    p = walk_forward_predictions(make_model, df, test_seasons)
    rows = [{"season": year, **score(df.loc[idx], p[idx])}
            for year, idx in df.loc[p.index].groupby("year").groups.items()]
    return pd.DataFrame(rows).set_index("season")


def compare(models: dict, df: pd.DataFrame, test_seasons=TEST_SEASONS) -> pd.DataFrame:
    """Backtest each model; add Brier skill relative to the model named 'grid'."""
    results = {name: backtest(make, df, test_seasons) for name, make in models.items()}
    return _with_brier_skill(pd.concat(results, names=["model", "season"]))


def holdout(models: dict, df: pd.DataFrame, seasons=HOLDOUT_SEASONS) -> pd.DataFrame:
    """Fit once on every season before the holdout, score each holdout season and all of them pooled."""
    train = df[df["year"] < min(seasons)]
    rows = []
    for name, make in models.items():
        model = make().fit(train)
        for label, test in [*((str(y), df[df["year"] == y]) for y in seasons),
                            ("all", df[df["year"].isin(seasons)])]:
            rows.append({"model": name, "season": label, **score(test, model.predict_proba(test))})
    return _with_brier_skill(pd.DataFrame(rows).set_index(["model", "season"]))


def _with_brier_skill(out: pd.DataFrame) -> pd.DataFrame:
    """Brier skill = 1 - Brier / Brier of the grid baseline in the same season."""
    grid = out.loc["grid", "brier"].reindex(out.index.get_level_values("season")).to_numpy()
    return out.assign(brier_skill=1 - out["brier"] / grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--holdout", action="store_true", help="also score the 2023-2024 holdout")
    args = parser.parse_args()
    pd.set_option("display.width", 160)

    df = build_features(build_base_table(load_tables(args.data_dir)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    results = compare(MODELS, df)
    results.to_csv(out / "backtest.csv")
    plot_backtest(results, figures / "backtest_log_loss.png")
    print(results.round(3).to_string())
    print("\nMean over test seasons:")
    print(results.groupby("model").mean().round(3).to_string())

    p = walk_forward_predictions(xgboost, df)
    table = reliability_table(df.loc[p.index, "podium"], p, RELIABILITY_BINS)
    table.to_csv(out / "reliability_xgb.csv")
    plot_reliability(table, figures / "reliability.png")
    print("\nReliability of XGBoost, pooled over test seasons:")
    print(table.round(3).to_string())

    scored = df.loc[p.index]
    gains = {"walk-forward": log_loss_gain(scored, p, walk_forward_predictions(GridBaseline, df))}

    if args.holdout:
        held = holdout(MODELS, df)
        held.to_csv(out / "holdout.csv")
        print(f"\nHoldout {HOLDOUT_SEASONS[0]}-{HOLDOUT_SEASONS[-1]}, trained on everything before:")
        print(held.round(3).to_string())

        train = df[df["year"] < min(HOLDOUT_SEASONS)]
        model, grid = xgboost().fit(train), GridBaseline().fit(train)
        for label, test in [*((str(y), df[df["year"] == y]) for y in HOLDOUT_SEASONS),
                            ("holdout pooled", df[df["year"].isin(HOLDOUT_SEASONS)])]:
            gains[label] = log_loss_gain(test, model.predict_proba(test), grid.predict_proba(test))

    gains = pd.DataFrame(gains).T
    gains.to_csv(out / "log_loss_gain.csv")
    print("\nXGBoost log-loss gain over the grid per race, 95% bootstrap CI over races:")
    print(gains.round(4).to_string())


if __name__ == "__main__":
    main()
