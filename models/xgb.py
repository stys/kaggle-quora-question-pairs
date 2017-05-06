# -*- coding: utf-8 -*-

"""
Train main XGBoost model
"""

import logging

from os.path import join as join_path

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split

from dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest, skfold, submission
from lib.utils import makedirs


def train_xgboost(train_df, test_df, skf, class_weight, **param):
    y = train_df[[FieldsTrain.is_duplicate]]
    train_df.drop([FieldsTrain.id, FieldsTrain.qid1, FieldsTrain.qid2, FieldsTrain.question1, FieldsTrain.question2, FieldsTrain.is_duplicate], axis=1, inplace=True)

    test_id = test_df[[FieldsTest.test_id]]
    test_df.drop([FieldsTest.test_id, FieldsTest.question1, FieldsTest.question2], axis=1, inplace=True)
    X = train_df.values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4231)

    sample_weight_train = np.vectorize(class_weight.get)(y_train)
    sample_weight_valid = np.vectorize(class_weight.get)(y_valid)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=sample_weight_valid)
    dtest = xgb.DMatrix(test_df.values)

    evallist = [(dvalid, 'eval'), (dtrain, 'train')]

    model = xgb.train(param, dtrain, param['num_round'], evallist)
    p_test = model.predict(dtest)
    submission('submission12.csv', test_id, p_test)


def main(conf):
    dump_dir = conf['xgboost.dump.dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df()

    logging.info('Loading test dataset')
    test_df = load_test_df()

    logging.info('Loading features')
    features = []
    for group, cnf in conf['xgboost.features'].iteritems():
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

    class_weight = {int(c['class']): c['weight'] for c in conf['weights']}

    train_xgboost(train_df, test_df, skfold(), class_weight, **conf['xgboost.param'])

if __name__ == '__main__':
    import project
    main(project.conf)