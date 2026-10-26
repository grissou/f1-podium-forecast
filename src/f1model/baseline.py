"""Benchmark 1: podium probability from starting grid slot alone.

This is the "market price" the model has to beat: it already knows that
the front row usually finishes on the podium.
"""

import numpy as np
import pandas as pd


class GridBaseline:
    """Historical podium rate per grid slot, with additive smoothing.

    Slots beyond `max_slot` share one pooled rate so sparse back-of-grid
    slots do not get noisy estimates.
    """

    def __init__(self, max_slot: int = 20, alpha: float = 1.0):
        self.max_slot = max_slot
        self.alpha = alpha

    def fit(self, df: pd.DataFrame) -> "GridBaseline":
        slot = self._slot(df["grid"])
        counts = df.groupby(slot)["podium"].agg(["sum", "count"])
        self.prior_ = df["podium"].mean()
        self.rates_ = (counts["sum"] + self.alpha * self.prior_) / (counts["count"] + self.alpha)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        slot = self._slot(df["grid"])
        return slot.map(self.rates_).fillna(self.prior_).to_numpy()

    def _slot(self, grid: pd.Series) -> pd.Series:
        return grid.clip(upper=self.max_slot + 1)
