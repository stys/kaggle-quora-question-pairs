# -*- coding: utf-8 -*-

"""
Compute feature set statistics
"""

import logging

from os.path import join as join_path

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from lib.utils import makedirs
from lib.dataset import load_train_df, load_test_df, skfold, FieldsTrain

from nbformat import write as nb_write
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_output


def main(conf):
    dump_dir = conf['exploration.dump.dir']
    makedirs(dump_dir)

    notebook_file = join_path(dump_dir, conf['exploration.dump.notebook'])
    notebook_cells = []

    images_dir = join_path(dump_dir, conf['exploration.dump.images.dir'])
    makedirs(images_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df()
    y = train_df[[FieldsTrain.is_duplicate]].values.flatten()

    logging.info('Loading test dataset')
    test_df = load_test_df()

    logging.info('Loading features')
    features = []
    for group, cnf in conf['features'].iteritems():
        logging.info('Loading features group: %s', group)

        features_dump_dir = cnf['dump']
        train_features_file = join_path(features_dump_dir, 'train.csv')
        test_features_file = join_path(features_dump_dir, 'test.csv')

        train_features = pd.read_csv(train_features_file)
        test_features = pd.read_csv(test_features_file)

        for fcnf in cnf['features']:
            feature = fcnf['feature']
            features.append(feature)
            train_col = fcnf.get('train_col', feature)
            test_col = fcnf.get('test_col', feature)
            train_df[feature] = train_features[train_col]
            test_df[feature] = test_features[test_col]

    figure = plt.figure(1, figsize=[8, 6])

    for feature in features:
        logging.info('Feature: %s', feature)
        train_stats = train_df[[feature]].describe()
        test_stats = test_df[[feature]].describe()

        cell = new_markdown_cell("# %s" % feature)
        notebook_cells.append(cell)

        sns.distplot(train_df[[feature]])
        sns.distplot(test_df[[feature]])

        image_file = join_path(images_dir, 'hist_%s.png' % feature)
        figure.savefig(image_file)

        plt.cla()

    nb = new_notebook(cells=notebook_cells)
    with open(notebook_file, 'w') as fh:
        nb_write(nb, fh)


if __name__ == '__main__':
    import project
    main(project.conf)
