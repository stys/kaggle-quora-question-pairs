# -*- coding: utf-8 -*-

"""
Shallow benchmark
https://www.kaggle.com/selfishgene/shallow-benchmark-0-31675-lb
"""

import json
import logging
from os.path import join as join_path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import logit
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from lib.project import project
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest, skfold
from lib.quality import reliability_curve
from lib.utils import makedirs, dump_config


def train_vectorizer(train_df, analyzer, ngram_range, min_df):
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english',
        binary=True,
        lowercase=True)

    vectorizer.fit(pd.concat((train_df.ix[:, Fields.question1], train_df.ix[:, Fields.question2])).unique())
    return vectorizer


def compute_feature_matrix(df, vectorizer, combine=None):

    fq1 = vectorizer.transform(df.ix[:, Fields.question1])
    fq2 = vectorizer.transform(df.ix[:, Fields.question2])

    if combine == 'diff':
        return abs(fq1 - fq2).tocsr()

    if combine == 'intersect':
        return fq1.multiply(fq2).tocsr()


def save_feature_matrix(X, filename):
    np.savez(filename, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)


def load_feature_matrix(filename):
    try:
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    except:
        return None


def train(X, y, skf, class_weight, **options):

    quality = dict(folds=[], full=dict())

    predictions = np.zeros(len(y))
    solver = options.get('solver') or 'lbfgs'
    penalty = options.get('penalty') or 'l2'
    alpha = options.get('alpha') or 1.0
    max_iter = options.get('max_iter') or 200
    random_state = options.get('seed') or None
    dump_dir = options.get('dump_dir') or '.'

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]

        dump_file = join_path(dump_dir, 'model_%d.pkl' % i)
        try:
            logging.info('Loading model for fold %d', i)
            f = joblib.load(dump_file)
        except:
            logging.info('Training model on fold %d', i)
            f = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter, random_state=random_state)
            f.fit(X_train, y_train)
            joblib.dump(f, dump_file)

        p_train = f.predict_proba(X_train)[:, 1]
        ll_train = log_loss(y_train, p_train)
        auc_train = roc_auc_score(y_train, p_train)

        logging.info('Train LL=%s AUC=%s', ll_train, auc_train)

        fpr_train, tpr_train, _ = roc_curve(y_train, p_train, pos_label=1)
        y_avg_train, p_avg_train = reliability_curve(y_train, p_train, nbins=50)

        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        p_valid = f.predict_proba(X_valid)[:, 1]
        ll_valid = log_loss(y_valid, p_valid)
        auc_valid = roc_auc_score(y_valid, p_valid)

        logging.info('Validation LL=%s AUC=%s', ll_valid, auc_valid)

        fpr_valid, tpr_valid, _ = roc_curve(y_valid, p_valid, pos_label=1)
        y_avg_valid, p_avg_valid = reliability_curve(y_valid, p_valid, nbins=50)

        predictions[valid_idx] = logit(p_valid)

        quality['folds'].append(dict(
            fold=i,
            dump=dump_file,
            ll=dict(train=ll_train, valid=ll_valid),
            auc=dict(train=auc_train, valid=auc_valid),
            roc=dict(
                train=dict(fpr=fpr_train.tolist(), tpr=tpr_train.tolist()),
                valid=dict(fpr=fpr_valid.tolist(), tpr=tpr_valid.tolist())
            ),
            reliability=dict(
                train=dict(y=y_avg_train.tolist(), p=p_avg_train.tolist()),
                valid=dict(y=y_avg_valid.tolist(), p=p_avg_valid.tolist())
            )
        ))

    # Train full model
    dump_file = join_path(dump_dir, 'model_full.pkl')

    try:
        logging.info('Loading full model')
        f_full = joblib.load(dump_file)
    except:
        logging.info('Training full model')
        f_full = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter, random_state=random_state)
        f_full.fit(X, y)
        joblib.dump(f_full, dump_file)

    p_full_train = f_full.predict_proba(X)[:, 1]
    ll_full_train = log_loss(y, p_full_train)
    auc_full_train = roc_auc_score(y, p_full_train)

    logging.info('Full LL=%s AUC=%s', ll_full_train, auc_full_train)

    quality['full']['unweighted'] = dict(
        dump=dump_file,
        ll=dict(train=ll_full_train),
        auc=dict(train=auc_full_train)
    )

    # Train full model with estimated class weights
    dump_file = join_path(dump_dir, 'model_full_weighted.pkl')

    try:
        logging.info('Loading full weighted model')
        f_full_weighted = joblib.load(dump_file)
    except:
        logging.info('Training full weighted model')
        f_full_weighted = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter,
                                         random_state=random_state, class_weight=class_weight)
        f_full_weighted.fit(X, y)
        joblib.dump(f_full_weighted, dump_file)

    p_full_train_weighted = f_full_weighted.predict_proba(X)[:, 1]
    sample_weight = np.vectorize(class_weight.get)(y)
    ll_full_train_weighted = log_loss(y, p_full_train_weighted, sample_weight=sample_weight)
    auc_full_train_weighted = roc_auc_score(y, p_full_train_weighted, sample_weight=sample_weight)

    quality['full']['weighted'] = dict(
        dump=dump_file,
        ll=dict(train=ll_full_train_weighted),
        auc=dict(train=auc_full_train_weighted)
    )

    return quality, predictions


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['dataset_raw'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['dataset_raw'])

    class_weight = {int(c['class']): c['weight'] for c in conf['weights']}

    for w, cnf in conf['linear'].iteritems():
        if not cnf.get_bool('enabled', True):
            continue

        if w == 'dataset':
            continue

        logging.info('Start training linear model: %s', w)

        dump_dir = cnf.get('dump.dir') or '.'
        makedirs(dump_dir)

        config_file = join_path(dump_dir, 'application.conf')
        dump_config(conf, config_file)

        vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
        quality_file = join_path(dump_dir, 'quality.json')

        y = train_df[FieldsTrain.is_duplicate]

        if cnf['dump.cache.enabled']:
            logging.info('Loading vectorizer')

            try:
                vectorizer = joblib.load(vectorizer_file)
            except:
                logging.info('Unable to load vectorizer')
                vectorizer = None

            if vectorizer is None:
                logging.info('Training vectorizer')

                vectorizer = train_vectorizer(train_df, **cnf['vectorizer'])
                nf = len(vectorizer.vocabulary_)
                logging.info('Feature count: %d', nf)

                logging.info('Dumping vectorizer')
                joblib.dump(vectorizer, vectorizer_file)

            features_cache_file = join_path(dump_dir, cnf['dump.cache.train'])
            logging.info('Loading cached train feature matrix from %s', features_cache_file)
            X = load_feature_matrix(features_cache_file)

            if X is None:
                logging.info('Unable to load cached train feature matrix')

                logging.info('Computing train feature matrix')
                X = compute_feature_matrix(train_df, vectorizer, combine=cnf['combine'])

                logging.info('Writing train feature matrix to %s', features_cache_file)
                save_feature_matrix(X, features_cache_file)
        else:
            logging.info('Training vectorizer')
            vectorizer = train_vectorizer(train_df, **cnf['vectorizer'])
            X = compute_feature_matrix(train_df, vectorizer, combine=cnf['combine'])
            nf = len(vectorizer.vocabulary_)
            logging.info('Feature count: %d', nf)

        logging.info('Training feature matrix: %s', X.shape)

        quality, predictions = train(X, y, skfold(), class_weight, dump_dir=dump_dir, **cnf['model'])

        with open(quality_file, 'w') as qfh:
            json.dump(quality, qfh)

        logging.info('Writing train set to disk')
        train_df[FieldsTrain.linear] = predictions
        train_df[[
            FieldsTrain.id,
            FieldsTrain.is_duplicate,
            FieldsTrain.linear
        ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

        if cnf['dump.cache.enabled']:
            features_cache_file = join_path(dump_dir, cnf['dump.cache.test'])

            logging.info('Loading cached test feature matrix from %s', features_cache_file)
            X = load_feature_matrix(features_cache_file)
            if X is None:
                logging.info('Unable to load cached test feature matrix')
                logging.info('Computing test feature matrix')
                X = compute_feature_matrix(test_df, vectorizer, combine=cnf['combine'])

                logging.info('Writing test feature matrix to cache')
                save_feature_matrix(X, features_cache_file)
        else:
            logging.info('Computing test feature matrix')
            X = compute_feature_matrix(test_df, vectorizer, combine=cnf['combine'])

        logging.info('Computing test predictions as average logit of cross-validation models')
        test_df[FieldsTest.linear_cv] = np.zeros(X.shape[0])
        for fold in quality['folds']:
            f = joblib.load(fold['dump'])
            p = logit(f.predict_proba(X)[:, 1])
            test_df[FieldsTest.linear_cv] = test_df[FieldsTest.linear_cv] + p
        test_df[FieldsTest.linear_cv] = test_df[FieldsTest.linear_cv] / len(quality['folds'])

        logging.info('Computing test predictions with full model')
        f = joblib.load(quality['full']['unweighted']['dump'])
        p = logit(f.predict_proba(X)[:, 1])
        test_df[FieldsTest.linear_full] = p

        logging.info('Computing test predictions with full weighted model')
        f = joblib.load(quality['full']['weighted']['dump'])
        p = logit(f.predict_proba(X)[:, 1])
        test_df[FieldsTest.linear_full_weighted] = p

        logging.info('Writing test set to disk')
        test_df[[
            FieldsTest.test_id,
            FieldsTest.linear_cv,
            FieldsTest.linear_full,
            FieldsTest.linear_full_weighted
        ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    main(project().conf)
