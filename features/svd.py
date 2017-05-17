# -*- coding: utf-8 -*-

"""
SVD features from sparce matrix of word and char ngrams
"""

import logging
from os.path import join as join_path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from scipy.sparse.linalg import svds
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

from lib.project import project
from lib.dataset import Fields, FieldsTrain, FieldsTest, load_train_df, load_test_df
from lib.utils import makedirs


def train_vectorizer(train_df, analyzer, ngram_range, min_df, binary):
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english',
        binary=binary,
        lowercase=True
    )

    vectorizer.fit(pd.concat((train_df.ix[:, Fields.question1], train_df.ix[:, Fields.question2])).unique())
    return vectorizer


def compute_feature_matrix(df, vectorizer, combine=None):
    fq1 = vectorizer.transform(df.ix[:, Fields.question1])
    fq2 = vectorizer.transform(df.ix[:, Fields.question2])

    combine = combine or 'diff'

    if combine == 'stack':
        return sparse_vstack([fq1, fq2])

    if combine == 'intersect':
        return fq1.multiply(fq2)

    if combine == 'diff':
        return abs(fq1 - fq2).tocsr()


def compute_svd(X, method, **options):
    if method == 'ARPACK':
        k = options.get('k', 10)
        maxiter = options.get('maxiter')
        tol = float(options.get('tol', 1.e-4))

        U, S, VT = svds(X, k=k, tol=tol, maxiter=maxiter, return_singular_vectors='vh')
        return S, VT


def save_feature_matrix(X, filename):
    np.savez(filename, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)


def load_feature_matrix(filename):
    try:
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    except:
        return None


def compute_svd_distance_eucl(row, fstem, ncomponents=10):
    dist = 0
    for j in xrange(ncomponents):
        fq1 = row[fstem + ('_%d_q1' % j)]
        fq2 = row[fstem + ('_%d_q2' % j)]
        dist += (fq1 - fq2) ** 2
    return dist


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['svd.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['svd.dataset'])

    for f, cnf in conf['svd'].iteritems():
        if f == 'dataset':
            continue

        if not cnf.get('enabled', True):
            continue

        logging.info('Start traning SVD model %s', f)

        dump_dir = cnf['dump.dir']
        makedirs(dump_dir)
        logging.info('Dump %s', dump_dir)

        vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
        try:
            logging.info('Loading vectorizer dump')
            vectorizer = joblib.load(vectorizer_file)
        except:
            logging.info('Loading vectorizer dump failed')
            logging.info('Traininig vectorizer: %s', cnf['vectorizer'])
            vectorizer = train_vectorizer(train_df, **cnf['vectorizer'])

            logging.info('Writing vectorizer dump')
            joblib.dump(vectorizer, vectorizer_file)

        train_features_matrix_file = join_path(dump_dir, 'train_features.npz')
        logging.info('Loading train features matrix')
        X = load_feature_matrix(train_features_matrix_file)
        if X is None:
            logging.info('Loading train feature matrix failed')
            logging.info('Computing train feature matrix')
            X = compute_feature_matrix(train_df, vectorizer, combine=cnf.get('model.transform', None))

            logging.info('Writing train feature matrix dump')
            save_feature_matrix(X, train_features_matrix_file)

        logging.info('Computing SVD decomposition')
        ksvd = cnf['model'].get_int('k')
        S, VT = compute_svd(X.asfptype(), **cnf['model'])
        Sinv = np.diag(1. / S) * np.sqrt(X.shape[0])
        logging.info('Singular values %s', S)

        logging.info('Computing train SVD features')
        U = X.dot(VT.transpose()).dot(Sinv)
        logging.info('Train features variance: %s', np.var(U, axis=0))

        features = map(lambda i: f + '_%d' % i, range(U.shape[1]))
        if cnf.get('model.transform', None) == 'stack':
            features_q1 = map(lambda s: s + '_q1', features)
            features_q2 = map(lambda s: s + '_q2', features)
            features = features_q1 + features_q2
            train_features_df_q1 = pd.DataFrame(U[:train_df.shape[0], :], columns=features_q1)
            train_features_df_q2 = pd.DataFrame(U[train_df.shape[0]:, :], columns=features_q2)
            train_df = pd.concat([train_df, train_features_df_q1, train_features_df_q2], axis=1)

            train_df['svd_dist_eucl'] = train_df.apply(lambda r: compute_svd_distance_eucl(r, f, ksvd), axis=1)
            features.append('svd_dist_eucl')
        else:
            train_features_df = pd.DataFrame(U, columns=features)
            train_df = pd.concat([train_df, train_features_df], axis=1)

        for feature in features:
            logging.info('Feature %s AUC=%s', feature, roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature]))

        logging.info('Writing train features dump')
        train_file = join_path(dump_dir, 'train.csv')
        train_df[[FieldsTrain.id, FieldsTrain.is_duplicate] + features].to_csv(train_file, index=False)

        test_features_matrix_file = join_path(dump_dir, 'test_features.npz')
        logging.info('Loading test features matrix')
        X = load_feature_matrix(test_features_matrix_file)
        if X is None:
            logging.info('Loading test feature matrix failed')
            logging.info('Computing test feature matrix')
            X = compute_feature_matrix(test_df, vectorizer, combine=cnf.get('model.transform', None))

            logging.info('Writing test feature matrix dump')
            save_feature_matrix(X, test_features_matrix_file)

        U = X.dot(VT.transpose()).dot(Sinv)
        logging.info('Test features variance: %s', np.var(U, axis=0))

        logging.info('Computing test SVD features')
        if cnf.get('model.transform', None) == 'stack':
            logging.info('Computing q1 test SVD features')
            test_features_df_q1 = pd.DataFrame(U[:test_df.shape[0], :], columns=features_q1)
            test_df = pd.concat([test_df, test_features_df_q1], axis=1)
            del test_features_df_q1

            logging.info('Computing q2 test SVD features')
            test_features_df_q2 = pd.DataFrame(U[test_df.shape[0]:, :], columns=features_q2)
            test_df = pd.concat([test_df, test_features_df_q2], axis=1)
            del test_features_df_q2

            logging.info('Computing svd distances')
            test_df['svd_dist_eucl'] = test_df.apply(lambda r: compute_svd_distance_eucl(r, f, ksvd), axis=1)
        else:
            test_features_df = pd.DataFrame(U, columns=features)
            test_df = pd.concat([test_df, test_features_df], axis=1)

        logging.info('Writing test features dump')
        test_file = join_path(dump_dir, 'test.csv')
        test_df[[FieldsTest.test_id] + features].to_csv(test_file, index=False)

if __name__ == '__main__':
    main(project().conf)
