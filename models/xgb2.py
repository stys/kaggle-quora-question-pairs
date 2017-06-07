# -*- coding: utf-8 -*-

"""
Train main XGBoost model
"""

from __future__ import division

import logging
from os.path import join as join_path

import heapq
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from lib.project import project
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest, submission
from lib.utils import makedirs, dump_config, json_string_config
from lib.quality import reliability_curve, rescale


def create_feature_map(features, feature_map_file):
    with open(feature_map_file, 'w') as fh:
        for i, feat in enumerate(features):
            fh.write('{0}\t{1}\tq\n'.format(i, feat))


def train_xgboost(X, y, w, **options):

    test_size = options.get('test_size') or 0.01
    random_state = options.get('seed') or 31

    X_train, X_valid, y_train, y_valid, w_train, w_valid, idx_train, idx_valid = \
        train_test_split(X, y, w, range(len(y)), test_size=test_size, random_state=random_state)

    logging.info('Train size: %s', X_train.shape)
    logging.info('Test size: %s', X_valid.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)

    eval_list = [(dtrain, 'train'), (dvalid, 'valid')]
    progress = dict()

    model = xgb.train(options, dtrain, options['num_round'], eval_list, evals_result=progress)

    p_train = model.predict(dtrain)
    p_valid = model.predict(dvalid)

    quality = dict(
        ll=dict(
            train=log_loss(y_train, p_train, sample_weight=w_train),
            valid=log_loss(y_valid, p_valid, sample_weight=w_valid)
        ),
        auc=dict(
            train=roc_auc_score(y_train, p_train, sample_weight=w_train),
            valid=roc_auc_score(y_valid, p_valid, sample_weight=w_valid)
        ),
        roc=dict(
            train=dict(zip(['fpr', 'tpr', 't'], roc_curve(y_train, p_train))),
            valid=dict(zip(['fpr', 'tpr', 't'], roc_curve(y_valid, p_valid)))
        ),
        reliability=dict(
            train=dict(zip(['avg_label', 'avg_pred'], reliability_curve(y_train, p_train, nbins=50, sample_weights=w_train))),
            valid=dict(zip(['avg_label', 'avg_pred'], reliability_curve(y_valid, p_valid, nbins=50, sample_weights=w_valid)))
        )
    )

    logging.info('Log-loss: %s', quality['ll'])

    quality['errors'] = dict(
        train=dict(
            type_i=heapq.nlargest(100, [(p, idx_train[j]) for j, p in enumerate(p_train) if y_train[j] == 0]),
            type_ii=heapq.nsmallest(100, [(p, idx_train[j]) for j, p in enumerate(p_train) if y_train[j] == 1])
        ),
        valid=dict(
            type_i=heapq.nlargest(100, [(p, idx_valid[j]) for j, p in enumerate(p_valid) if y_valid[j] == 0]),
            type_ii=heapq.nsmallest(100, [(p, idx_valid[j]) for j, p in enumerate(p_valid) if y_valid[j] == 1])
        )
    )

    return model, progress, quality


def plot_progress(progress, img_dir):
    fig1 = plt.figure(1, figsize=[8, 4])
    plt.plot(progress['train']['logloss'], 'b-', label='train')
    plt.plot(progress['valid']['logloss'], 'r-', label='valid')
    plt.legend()
    plt.grid()
    plt.title('log loss')
    fig1.savefig(join_path(img_dir, 'learn_curves_ll.png'))

    fig2 = plt.figure(2, figsize=[8, 4])
    plt.plot(progress['train']['auc'], 'b-', label='train')
    plt.plot(progress['valid']['auc'], 'r-', label='valid')
    plt.legend()
    plt.grid()
    plt.title('AUC')
    fig2.savefig(join_path(img_dir, 'lean_curves_auc.png'))


def plot_quality(quality, img_dir):
    fig1 = plt.figure(3, figsize=[8, 4])
    plt.plot(quality['roc']['train']['fpr'], quality['roc']['train']['tpr'], 'b-', label='train')
    plt.plot(quality['roc']['valid']['fpr'], quality['roc']['valid']['tpr'], 'r-', label='valid')
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k--')
    plt.legend()
    plt.grid()
    plt.title('ROC')
    fig1.savefig(join_path(img_dir, 'roc.png'))

    fig2 = plt.figure(4, figsize=[8, 4])
    plt.plot(quality['reliability']['train']['avg_pred'], quality['reliability']['train']['avg_label'], 'b-', label='train')
    plt.plot(quality['reliability']['valid']['avg_pred'], quality['reliability']['valid']['avg_label'], 'r-', label='valid')
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k--')
    plt.legend()
    plt.grid()
    plt.title('Reliability')
    fig2.savefig(join_path(img_dir, 'reliability.png'))


def main(conf):
    dump_dir = conf['xgboost.dump.dir']
    makedirs(dump_dir)

    dump_config_file = join_path(dump_dir, 'application.conf')
    dump_config(conf, dump_config_file)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['xgboost.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['xgboost.dataset'])

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

    train_df_flipped = train_df.copy()
    for flip in conf['flip']:
        train_df_flipped[flip[0]] = train_df[[flip[1]]]
        train_df_flipped[flip[1]] = train_df[[flip[0]]]

    train_df = pd.concat([train_df, train_df_flipped], axis=0, ignore_index=True)
    logging.info('Train dataset: %s', train_df.shape)

    y = train_df[[FieldsTrain.is_duplicate]].values.flatten()
    logging.info('Train dataset CTR: %s', y.sum() / len(y))

    class_weight = {int(c['class']): c['weight'] for c in conf['weights']}
    w = np.vectorize(class_weight.get)(y)
    logging.info('Train dataset weighted CTR: %s', sum(y * w) / sum(w))

    q1 = train_df[Fields.question1].values
    q2 = train_df[Fields.question2].values

    train_df.drop([
        FieldsTrain.id,
        FieldsTrain.qid1,
        FieldsTrain.qid2,
        FieldsTrain.question1,
        FieldsTrain.question2,
        FieldsTrain.is_duplicate], axis=1, inplace=True)

    X = train_df.values

    logging.info('Training XGBoost model')
    model, progress, quality = train_xgboost(X, y, w, **conf['xgboost.param'])

    logging.info('Writing model dump')
    model_dump_file = join_path(dump_dir, 'model_dump.txt')
    model.dump_model(model_dump_file, fmap=feature_map_file, with_stats=True)
    model_file = join_path(dump_dir, 'model.bin')
    model.save_model(model_file)

    logging.info('Writing quality')
    # plot_quality(quality, dump_dir)

    logging.info('Writing top errors')
    errors_file = join_path(dump_dir, 'errors.csv')
    with open(errors_file, 'w') as fh:
        fh.write('y,p,question1,question2,sample\n')
        for e in quality['errors']['train']['type_i']:
            fh.write('%d,%s,%s,%s,%s\n' % (0, e[0], q1[e[1]], q2[e[1]], 'train'))
        for e in quality['errors']['train']['type_ii']:
            fh.write('%d,%s,%s,%s,%s\n' % (1, e[0], q1[e[1]], q2[e[1]], 'train'))
        for e in quality['errors']['valid']['type_i']:
            fh.write('%d,%s,%s,%s,%s\n' % (0, e[0], q1[e[1]], q2[e[1]], 'valid'))
        for e in quality['errors']['valid']['type_ii']:
            fh.write('%d,%s,%s,%s,%s\n' % (1, e[0], q1[e[1]], q2[e[1]], 'valid'))

    logging.info('Writing progress file')
    # plot_progress(progress, dump_dir)
    progress_file = join_path(dump_dir, 'progress.json')
    with open(progress_file, 'w') as fh:
        json.dump(progress, fh)

    logging.info('Writing feature scores')
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
    main(project().conf)
