# -*- coding: utf-8 -*-

"""
Train shallow neural network on SVD features
"""

import logging

from os.path import join as join_path

from lib.project import project
from lib.utils import makedirs
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest, skfold


def main(conf):
    dump_dir = conf['shallow']['dump']['dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['shallow']['dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['shallow']['dataset'])




if __name__ == '__main__':
    main(project().conf)
