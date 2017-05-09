# -*- coding: utf-8 -*-

from lib.dataset import load_train_df, load_test_df


def test_load():
    train_df = load_train_df()
    assert train_df.shape == (404290, 6)

    test_df = load_test_df()
    assert test_df.shape == (2345796, 3)
