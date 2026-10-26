import numpy as np

from f1model.explain import shap_values
from f1model.features import FEATURES, build_features


def test_shap_values_cover_every_test_row(synthetic_base):
    df = build_features(synthetic_base)
    values, base = shap_values(df, test_seasons=[2012])
    assert list(values.columns) == FEATURES
    assert len(values) == (df["year"] == 2012).sum()
    assert np.isfinite(values.to_numpy()).all()
    assert base.index.equals(values.index)
