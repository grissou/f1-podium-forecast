import numpy as np
import pandas as pd

from f1model.baseline import GridBaseline


def test_front_of_grid_scores_higher_than_back():
    train = pd.DataFrame({
        "grid":   [1, 1, 1, 1, 10, 10, 10, 10],
        "podium": [1, 1, 1, 0, 0, 0, 0, 1],
    })
    model = GridBaseline(alpha=1.0).fit(train)
    p = model.predict_proba(pd.DataFrame({"grid": [1, 10]}))
    assert p[0] > p[1]
    assert np.all((p > 0) & (p < 1))


def test_unseen_slot_falls_back_to_overall_rate():
    train = pd.DataFrame({"grid": [1, 2], "podium": [1, 0]})
    model = GridBaseline().fit(train)
    p = model.predict_proba(pd.DataFrame({"grid": [7]}))
    assert p[0] == train["podium"].mean()


def test_back_of_grid_slots_are_pooled():
    train = pd.DataFrame({"grid": [21, 22, 25], "podium": [0, 0, 1]})
    model = GridBaseline(max_slot=20).fit(train)
    p = model.predict_proba(pd.DataFrame({"grid": [21, 24]}))
    assert p[0] == p[1]
