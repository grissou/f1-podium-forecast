import numpy as np

from f1model.features import build_features
from f1model.train import XGB_DEFAULTS, XGB_SEARCH_SPACE, XGBoostModel


def test_xgboost_tunes_on_last_training_season(synthetic_base):
    df = build_features(synthetic_base)
    train, test = df[df["year"] < 2012], df[df["year"] == 2012]
    model = XGBoostModel(n_trials=2).fit(train)

    p = model.predict_proba(test)
    assert p.shape == (len(test),)
    assert np.all((p > 0) & (p < 1))
    assert 1 <= model.best_rounds_ <= 1000
    assert model.best_params_ == XGB_DEFAULTS or all(
        model.best_params_[k] in XGB_SEARCH_SPACE[k] for k in XGB_SEARCH_SPACE
    )
    assert np.isfinite(model.val_log_loss_)
