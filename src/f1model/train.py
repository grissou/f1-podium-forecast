"""Models that share the GridBaseline interface: fit(df) and predict_proba(df)."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from f1model.features import FEATURES


class FeatureModel:
    """Wrap an sklearn estimator so it trains on FEATURES and predicts P(podium)."""

    def __init__(self, estimator, features=FEATURES):
        self.estimator = estimator
        self.features = list(features)

    def fit(self, df: pd.DataFrame) -> "FeatureModel":
        self.estimator.fit(df[self.features], df["podium"])
        return self

    def predict_proba(self, df: pd.DataFrame):
        return self.estimator.predict_proba(df[self.features])[:, 1]


def logistic_regression() -> FeatureModel:
    """Benchmark 2: linear model on the same features. Imputer and scaler fit per training window."""
    return FeatureModel(make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
    ))


# Shallow trees, slow learning, row/column subsampling: the usual recipe for a few
# thousand rows where the aim is a calibrated probability rather than a leaderboard.
XGB_DEFAULTS = dict(
    max_depth=3, learning_rate=0.05, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
)
XGB_SEARCH_SPACE = dict(
    max_depth=[2, 3, 4, 5], learning_rate=[0.02, 0.05, 0.1], min_child_weight=[1, 5, 10, 20],
    subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], reg_lambda=[1.0, 5.0, 10.0],
)
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50


class XGBoostModel(FeatureModel):
    """Gradient boosting tuned on the last training season, then refit on the whole window.

    The last season of the training window is the inner validation set: it picks
    the number of boosting rounds (early stopping) and, with `n_trials > 0`,
    the best of `n_trials` random draws from XGB_SEARCH_SPACE by log loss.
    The outer test season is never seen. Candidate 0 is always XGB_DEFAULTS.
    """

    def __init__(self, n_trials: int = 0, seed: int = 0, features=FEATURES):
        super().__init__(estimator=None, features=features)
        self.n_trials = n_trials
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "XGBoostModel":
        last_season = df["year"].max()
        inner_train, inner_val = df[df["year"] < last_season], df[df["year"] == last_season]
        if inner_train.empty:
            raise ValueError("XGBoostModel needs at least two seasons: the last one is held out for early stopping")
        candidates = [XGB_DEFAULTS] + list(
            ParameterSampler(XGB_SEARCH_SPACE, n_iter=self.n_trials, random_state=self.seed)
        )

        best = None
        for params in candidates:
            probe = XGBClassifier(
                **params, n_estimators=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=self.seed,
            ).fit(
                inner_train[self.features], inner_train["podium"],
                eval_set=[(inner_val[self.features], inner_val["podium"])], verbose=False,
            )
            val_loss = log_loss(inner_val["podium"], probe.predict_proba(inner_val[self.features])[:, 1])
            if best is None or val_loss < best[0]:
                best = (val_loss, params, probe.best_iteration + 1)

        self.val_log_loss_, self.best_params_, self.best_rounds_ = best
        self.estimator = XGBClassifier(
            **self.best_params_, n_estimators=self.best_rounds_, random_state=self.seed
        )
        return super().fit(df)


def xgboost() -> XGBoostModel:
    return XGBoostModel(n_trials=0)


def xgboost_tuned() -> XGBoostModel:
    return XGBoostModel(n_trials=20)
