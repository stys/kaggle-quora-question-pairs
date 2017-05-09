# -*- coding: utf-8 -*-

"""
Train main XGBoost model
"""

from __future__ import division

import logging
from os.path import join as join_path

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from lib.dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest, submission
from lib.utils import makedirs, dump_config, json_string_config


def create_feature_map(features, feature_map_file):
    with open(feature_map_file, 'w') as fh:
        for i, feat in enumerate(features):
            fh.write('{0}\t{1}\tq\n'.format(i, feat))


def train_xgboost(X, y, w, **options):
    test_size = options.get('test_size') or 0.2
    random_state = options.get('seed') or 0

    X_train, X_valid, y_train, y_valid, w_train, w_valid = \
        train_test_split(X, y, w, test_size=test_size, random_state=random_state)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)

    eval_list = [(dtrain, 'train'), (dvalid, 'valid')]
    progress = dict()

    model = xgb.train(options, dtrain, options['num_round'], eval_list, evals_result=progress)
    return model, progress


def plot_progress(progress, img_dir):
    fig1 = plt.figure(1, figsize=[8, 4])
    plt.plot(progress['train']['logloss'], '-b', label='train')
    plt.plot(progress['valid']['logloss'], '-r', label='valid')
    plt.legend()
    plt.grid()
    plt.title('log loss')
    fig1.savefig(join_path(img_dir, 'learn_curves_ll.png'))

    fig2 = plt.figure(2, figsize=[8, 4])
    plt.plot(progress['train']['auc'], '-b', label='train')
    plt.plot(progress['valid']['auc'], '-r', label='valid')
    plt.legend()
    plt.grid()
    plt.title('AUC')
    fig2.savefig(join_path(img_dir, 'lean_curves_auc.png'))


def plot_score(score, img_dir):
    pass


def main(conf):
    dump_dir = conf['xgboost.dump.dir']
    makedirs(dump_dir)

    dump_config_file = join_path(dump_dir, 'application.conf')
    dump_config(conf, dump_config_file)

    logging.info('Loading train dataset')
    train_df = load_train_df()

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

    feature_map_file = join_path(dump_dir, 'xgb.fmap')
    create_feature_map(features, feature_map_file)

    y = train_df[[FieldsTrain.is_duplicate]].values.flatten()
    logging.info('Train dataset CTR: %s', y.sum() / len(y))

    class_weight = {int(c['class']): c['weight'] for c in conf['weights']}
    w = np.vectorize(class_weight.get)(y)
    logging.info('Train dataset weighted CTR: %s', sum(y * w) / sum(w))

    train_df.drop([
        FieldsTrain.id,
        FieldsTrain.qid1,
        FieldsTrain.qid2,
        FieldsTrain.question1,
        FieldsTrain.question2,
        FieldsTrain.is_duplicate], axis=1, inplace=True)

    X = train_df.values

    logging.info('Training XGBoost model: %s', json_string_config(conf['xgboost.param']))
    model, progress = train_xgboost(X, y, w, **conf['xgboost.param'])

    logging.info('Writing model dump')
    model_dump_file = join_path(dump_dir, 'model_dump.txt')
    model.dump_model(model_dump_file, fmap=feature_map_file, with_stats=True)
    model_file = join_path(dump_dir, 'model.bin')
    model.save_model(model_file)

    logging.info('Writing progress file')
    progress_file = join_path(dump_dir, 'progress.json')
    with open(progress_file, 'w') as fh:
        json.dump(progress, fh)
    plot_progress(progress, dump_dir)

    score_weight = model.get_score(fmap=feature_map_file, importance_type='weight')
    score_gain = model.get_score(fmap=feature_map_file, importance_type='gain')
    score_cover = model.get_score(fmap=feature_map_file, importance_type='cover')
    split_histograms = dict()
    for f in features:
        split_histograms[f] = model.get_split_value_histogram(f, fmap=feature_map_file)

    scores = pd.DataFrame([score_weight, score_gain, score_cover]).transpose()
    scores.index.name = 'feature'
    scores.rename(columns={0: 'weight', 1: 'gain', 2: 'cover'}, inplace=True)
    weight_total = scores['weight'].sum()
    scores['weight'] = scores['weight'] / weight_total
    scores.sort_values(by='weight', ascending=False, inplace=True)
    scores.to_csv(join_path(dump_dir, 'feature_scores.csv'))

    logging.info('Computing test predictions')
    test_ids = test_df[[FieldsTest.test_id]]
    test_df.drop([FieldsTest.test_id, FieldsTest.question1, FieldsTest.question2], axis=1, inplace=True)
    dtest = xgb.DMatrix(test_df.values)
    p_test = model.predict(dtest)

    logging.info('Writing submission file')
    submission_file = join_path(dump_dir, 'submission.csv')
    submission(submission_file, test_ids, p_test)

if __name__ == '__main__':
    import project
    main(project.conf)
