# -*- coding: utf-8 -*-

"""
Shallow benchmark
https://www.kaggle.com/selfishgene/shallow-benchmark-0-31675-lb
"""

import logging

import errno

from os.path import join as join_path
from os import makedirs

import json

import numpy as np
import pandas as pd

from scipy.special import logit
from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from dataset import load_train_df, load_test_df, Fields, FieldsTrain, skfold
from lib.quality import reliability_curve


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


def compute_feature_matrix(df, vectorizer):
    fq1 = vectorizer.transform(df.ix[:, Fields.question1])
    fq2 = vectorizer.transform(df.ix[:, Fields.question2])
    return abs(fq1 - fq2).tocsr()


def save_feature_matrix(X, filename):
    np.savez(filename, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)


def load_feature_matrix(filename):
    try:
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    except:
        return None


def train(X, y, skf, class_weight, **options):

    quality = dict(vectorizer=None, folds=[], full=dict())

    predictions = np.zeros(len(y))
    solver = options.get('solver') or 'lbfgs'
    penalty = options.get('penalty') or 'l2'
    alpha = options.get('alpha') or 1.0
    max_iter = options.get('max_iter') or 200
    random_state = options.get('seed') or None
    dump_dir = options.get('dump_dir') or '.'

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        logging.info('Training model on fold %d', i)

        f = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter, random_state=random_state)
        X_train = X[train_idx]
        y_train = y[train_idx]
        f.fit(X_train, y_train)

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
        auc_valid = log_loss(y_valid, p_valid)

        logging.info('Validation LL=%s AUC=%s', ll_valid, auc_valid)

        fpr_valid, tpr_valid, _ = roc_curve(y_valid, p_valid, pos_label=1)
        y_avg_valid, p_avg_valid = reliability_curve(y_valid, p_valid, nbins=50)

        predictions[valid_idx] = logit(p_valid)

        dump_file = join_path(dump_dir, 'model_%d.pkl' % i)
        joblib.dump(f, dump_file)

        quality['folds'].append(dict(
            fold=i,
            dump=dump_file,
            ll=dict(train=ll_train, valid=ll_valid),
            auc=dict(train=auc_train, valid=auc_valid),
            roc=dict(
                train=dict(fpr=fpr_train, tpr=tpr_train),
                valid=dict(fpr=fpr_valid, tpr=tpr_valid)
            ),
            reliability=dict(
                train=dict(y=y_avg_train, p=p_avg_train),
                valid=dict(y=y_avg_valid, p=p_avg_valid)
            )
        ))

    # Train full model
    logging.info('Training full model')
    f_full = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter, random_state=random_state)
    f_full.fit(X, y)

    p_full_train = f_full.predict_proba(X)[:, 1]
    ll_full_train = log_loss(y, p_full_train)
    auc_full_train = log_loss(y, p_full_train)

    logging.info('Full LL=%s AUC=%s', ll_full_train, auc_full_train)

    dump_file = join_path(dump_dir, 'model_full.pkl')
    joblib.dump(f_full, dump_file)

    quality['full']['unweighted'] = dict(
        dump=dump_file,
        ll=dict(train=ll_full_train),
        auc=dict(train=auc_full_train)
    )

    # Train full model with estimated class weights
    logging.info('Training full weighted model')
    f_full_weighted = LogisticRegression(solver=solver, penalty=penalty, C=alpha, max_iter=max_iter,
                                         random_state=random_state, class_weight=class_weight)
    f_full_weighted.fit(X, y)

    p_full_train_weighted = f_full_weighted.predict_proba(X)[:, 1]
    sample_weight = np.vectorize(class_weight.get)(y)
    ll_full_train_weighted = log_loss(y, p_full_train_weighted, sample_weight=sample_weight)
    auc_full_train_weighted = roc_auc_score(y, p_full_train_weighted, sample_weight=sample_weight)

    dump_file = join_path(dump_dir, 'model_full_weighted.pkl')
    joblib.dump(f_full_weighted, dump_file)

    quality['full']['weighted'] = dict(
        dump=dump_file,
        ll=dict(train=ll_full_train_weighted),
        auc=dict(train=auc_full_train_weighted)
    )

    return quality, predictions


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df()

    logging.info('Loading test dataset')
    test_df = load_test_df()

    class_weight = {int(c['class']): c['weight'] for c in conf['weights']}

    for w, cnf in conf['linear'].iteritems():
        dump_dir = cnf.get('dump.dir') or '.'

        try:
            makedirs(dump_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
        quality_file = join_path(dump_dir, 'quality.json')

        y = np.array(train_df[[FieldsTrain.is_duplicate]])

        if cnf['dump.cache.enabled']:
            logging.info('Loading cached train feature matrix')

            try:
                vectorizer = joblib.load(vectorizer_file)
            except:
                vectorizer = None

            X = load_feature_matrix(join_path(dump_dir, cnf['dump.cache.train']))

            if vectorizer is None or X is None:
                logging.info('Unable to load cached train feature matrix')
                logging.info('Training vectorizer')

                vectorizer = train_vectorizer(train_df, **cnf['vectorizer'])
                nf = len(vectorizer.vocabulary_)
                logging.info('Feature count: %d', nf)

                logging.info('Dumping vectorizer')
                joblib.dump(vectorizer, vectorizer_file)

                logging.info('Computing train feature matrix')
                X = compute_feature_matrix(train_df, vectorizer)

                logging.info('Writing train feature matrix to cache')
                save_feature_matrix(X, join_path(dump_dir, cnf['dump.cache.train']))
        else:
            logging.info('Training vectorizer')
            vectorizer = train_vectorizer(train_df, **cnf['vectorizer'])
            X = compute_feature_matrix(train_df, vectorizer)
            nf = len(vectorizer.vocabulary_)
            logging.info('Feature count: %d', nf)

        quality, predictions = train(X, y, skfold(), class_weight, dump_dir=dump_dir, **cnf['model'])
        json.dump(quality, quality_file)

        logging.info('Writing train set to disk')
        train_df['linear'] = predictions
        train_df.to_csv(join_path(dump_dir, 'train.csv'))

        if cnf['dump.cache.enabled']:
            logging.info('Loading cached test feature matrix')
            X = load_feature_matrix(join_path(dump_dir, cnf['dump.cache.test']))
            if X is None:
                logging.info('Unable to load cached test feature matrix')
                logging.info('Computing test feature matrix')
                X = compute_feature_matrix(test_df, vectorizer)

                logging.info('Writing test feature matrix to cache')
                save_feature_matrix(X, cnf['dump.cache.test'])
        else:
            logging.info('Computing test feature matrix')
            X = compute_feature_matrix(test_df, vectorizer)

        logging.info('Computing test predictions as average logit of cross-validation models')
        for fold in quality['folds']:
            f = joblib.load(fold['dump'])
            p = logit(f.predict_proba(X)[:, 1])
            test_df['linear_cv'] = test_df['linear_cv'] + p
        test_df['linear_cv'] /= len(quality['folds'])

        logging.info('Computing test predictions with full model')
        f = joblib.load(quality['full']['unweighted']['dump'])
        p = logit(f.predict_proba(X)[:, 1])
        test_df['linear_full'] = p

        logging.info('Computing test predictions with full weighted model')
        f = joblib.load(quality['full']['weighted']['dump'])
        p = logit(f.predict_proba(X)[:, 1])
        test_df['linear_full_weighted'] = p

        logging.info('Writing test set to disk')
        test_df.to_csv(join_path(dump_dir, 'test.csv'))

if __name__ == '__main__':
    import project
    main(project.conf)
