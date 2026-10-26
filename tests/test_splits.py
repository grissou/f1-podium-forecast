from f1model.splits import walk_forward


def test_train_is_strictly_before_test_and_grows(synthetic_base):
    df = synthetic_base
    folds = list(walk_forward(df, test_seasons=[2011, 2012]))
    assert [year for year, _, _ in folds] == [2011, 2012]
    for year, train, test in folds:
        assert df.loc[train, "year"].max() < year
        assert (df.loc[test, "year"] == year).all()
        assert df.loc[train, "date"].max() < df.loc[test, "date"].min()
    assert len(folds[1][1]) > len(folds[0][1])
