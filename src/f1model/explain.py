"""SHAP attribution of the out-of-sample forecasts, and the figures in the README.

Usage: python -m f1model.explain --data-dir data/raw

SHAP values are computed on each walk-forward test season from the model fitted
on the seasons before it, so the explanations are as out-of-sample as the
forecasts. Values are in log-odds: +0.7 roughly doubles the odds of a podium.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from f1model.dataset import build_base_table  # noqa: E402
from f1model.features import FEATURES, build_features  # noqa: E402
from f1model.load import load_tables  # noqa: E402
from f1model.splits import TEST_SEASONS, walk_forward  # noqa: E402
from f1model.train import xgboost  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def shap_values(df: pd.DataFrame, test_seasons=TEST_SEASONS) -> tuple[pd.DataFrame, pd.Series]:
    """Out-of-sample SHAP values for every test-season row, plus each fold's base value (log-odds)."""
    values, bases = [], []
    for _, train, test in walk_forward(df, test_seasons):
        explainer = shap.TreeExplainer(xgboost().fit(df.loc[train]).estimator)
        values.append(pd.DataFrame(explainer.shap_values(df.loc[test, FEATURES]), index=test, columns=FEATURES))
        bases.append(pd.Series(float(np.ravel(explainer.expected_value)[0]), index=test))
    return pd.concat(values), pd.concat(bases)


def plot_backtest(results: pd.DataFrame, path: Path) -> None:
    """Log loss per test season for the three main models (results = evaluate.compare output)."""
    names = {"grid": "Grid baseline", "logistic": "Logistic regression", "xgb": "XGBoost"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (key, label), color in zip(names.items(), COLORS):
        s = results.loc[key, "log_loss"]
        ax.plot(s.index, s.values, color=color, lw=2, marker="o", ms=5, label=label)
        ax.text(s.index[-1] + 0.15, s.values[-1], label, color=color, fontsize=8, va="center")
    ax.set_xticks(s.index)
    ax.set_xlim(s.index[0] - 0.3, s.index[-1] + 1.8)
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Out-of-sample log loss by test season")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(table: pd.DataFrame, path: Path) -> None:
    """Forecast probability vs observed podium rate per bin; dot area is the number of forecasts."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    size = 30 + 400 * table["n"] / table["n"].max()
    ax.scatter(table["predicted"], table["observed"], s=size, color=COLORS[0], alpha=0.85, edgecolor="white")
    for row in table.itertuples():
        ax.annotate(f"n={row.n}", (row.predicted, row.observed), xytext=(8, -4),
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("forecast P(podium)")
    ax.set_ylabel("observed podium rate")
    ax.set_title("Reliability of XGBoost forecasts")
    fig.savefig(path)
    plt.close(fig)


def plot_importance(values: pd.DataFrame, path: Path) -> None:
    importance = values.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(importance.index, importance.values, color=COLORS[0], height=0.65)
    for i, v in enumerate(importance.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP|  (log-odds)")
    ax.set_title("What moves the podium forecast, 2016-2022 out of sample")
    fig.savefig(path)
    plt.close(fig)


def plot_by_season(values: pd.DataFrame, season: pd.Series, path: Path, top: int = 6) -> None:
    by_season = values.abs().groupby(season).mean()
    features = by_season.mean().sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for feature, color in zip(features, COLORS):
        ax.plot(by_season.index, by_season[feature], color=color, lw=2, marker="o", ms=5, label=feature)
    ax.set_ylabel("mean |SHAP|  (log-odds)")
    ax.set_title("Feature influence by test season")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def plot_waterfall(df: pd.DataFrame, values: pd.DataFrame, base: pd.Series, row, path: Path) -> None:
    """One forecast taken apart: how each feature moved this driver's log-odds from the base rate."""
    explanation = shap.Explanation(
        values=values.loc[row].to_numpy(), base_values=base[row],
        data=df.loc[row, FEATURES].to_numpy().astype(float), feature_names=FEATURES,
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    tables = load_tables(args.data_dir)
    df = build_features(build_base_table(tables))
    values, base = shap_values(df)
    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    importance = values.abs().mean().sort_values(ascending=False)
    importance.to_csv(out / "shap_importance.csv", header=["mean_abs_shap"])
    print("Mean |SHAP| per feature, 2016-2022 out of sample:")
    print(importance.round(3).to_string())

    plot_importance(values, figures / "shap_importance.png")
    plot_by_season(values, df.loc[values.index, "year"], figures / "shap_by_season.png")

    # The podium from furthest back on the grid is the most instructive single forecast.
    scored = df.loc[values.index]
    row = scored[scored["podium"] == 1]["grid"].idxmax()
    plot_waterfall(df, values, base, row, figures / "shap_waterfall.png")
    driver = tables["drivers"].set_index("driverId").loc[df.loc[row, "driverId"]]
    race = tables["races"].set_index("raceId").loc[df.loc[row, "raceId"], "name"]
    print(f"\nWaterfall: {driver['forename']} {driver['surname']}, {int(df.loc[row, 'year'])} {race}, "
          f"grid {int(df.loc[row, 'grid'])} -> finished {int(df.loc[row, 'position_order'])}, "
          f"forecast P(podium) = {1 / (1 + np.exp(-(base[row] + values.loc[row].sum()))):.2f}")


if __name__ == "__main__":
    main()
