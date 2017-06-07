# -*- coding: utf-8 -*-

"""
Train shallow neural network on SVD features
"""

import logging

from os.path import join as join_path

import numpy as np

from scipy.special import logit

from lib.project import project
from lib.utils import makedirs
from lib.dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest, skfold
from lib.quality import reliability_curve
from lib.utils import dump_config

from features.linear import load_feature_matrix, save_feature_matrix

from sklearn.externals import joblib
from sklearn.utils.extmath import safe_sparse_dot

from svd import train_vectorizer, compute_feature_matrix, compute_svd

from tqdm import tqdm


def euclidean(Uq1, Uq2):
    return np.sqrt(np.sum(np.multiply(Uq1 - Uq2, Uq1 - Uq2), axis=1))


def cosine(Uq1, Uq2):
    norm_q1 = np.sqrt(np.sum(np.multiply(Uq1, Uq1), axis=1)) + 1.e-9
    norm_q2 = np.sqrt(np.sum(np.multiply(Uq2, Uq2), axis=1)) + 1.e-9
    return np.sum(np.multiply(Uq1, Uq2), axis=1) / norm_q1 / norm_q2


def main(conf):
    dump_dir = conf['svdres.dump.dir']
    makedirs(dump_dir)

    dump_config_file = join_path(dump_dir, 'application.conf')
    dump_config(conf, dump_config_file)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['svdres.dataset'])

    vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
    try:
        logging.info('Loading vectorizer dump')
        vectorizer = joblib.load(vectorizer_file)
    except:
        logging.info('Loading vectorizer dump failed')
        logging.info('Traininig vectorizer')
        vectorizer = train_vectorizer(train_df, **conf['svdres.vectorizer'])

        logging.info('Writing vectorizer dump')
        joblib.dump(vectorizer, vectorizer_file)

    features_file = join_path(dump_dir, 'features_train.npz')
    logging.info('Loading cached train feature matrix from %s', features_file)
    X = load_feature_matrix(features_file)

    if X is None:
        logging.info('Unable to load cached train feature matrix')

        logging.info('Computing train feature matrix')
        X = compute_feature_matrix(train_df, vectorizer, combine='stack')

        logging.info('Writing train feature matrix to %s', features_file)
        save_feature_matrix(X, features_file)

    logging.info('Loading SVD decomposition')
    k = conf['svdres.svd'].get_int('k')
    singular_values_file = join_path(dump_dir, 'singular_values.txt')
    singular_vectors_file = join_path(dump_dir, 'singular_vectors.npz')
    try:
        S = np.loadtxt(singular_values_file)
        VT = np.load(singular_vectors_file)['VT']
        assert k == len(S)
    except:
        logging.info('Loading SVD decomposition failed')
        logging.info('Computing SVD decomposition')
        S, VT = compute_svd(X.asfptype(), **conf['svdres.svd'])

        logging.info('Writing singular values to file')
        np.savetxt(singular_values_file, S)
        np.savez(singular_vectors_file, VT=VT)

    logging.info('Train matrix %s', X.shape)
    logging.info('Computing train SVD residuals')
    L = X.shape[0] / 2
    Xq1 = X[:L, :]
    Xq2 = X[L:, :]

    start = 0
    batch = 100
    eucl = np.zeros(Xq1.shape[0])
    cos = np.zeros(Xq1.shape[0])
    q1res = np.zeros(Xq1.shape[0])
    q2res = np.zeros(Xq1.shape[0])
    while start < Xq1.shape[0]:
        finish = min(start + batch, Xq1.shape[0])

        Xq1_batch = Xq1[start:finish, :]
        nq1 = (Xq1_batch.multiply(Xq1_batch)).sum(axis=1).flatten()

        Rq1 = safe_sparse_dot(Xq1_batch, VT.transpose()).dot(VT) - Xq1_batch
        nrq1 = np.sum(np.multiply(Rq1, Rq1), axis=1).flatten()

        Xq2_batch = Xq2[start:finish, :]
        nq2 = (Xq2_batch.multiply(Xq2_batch)).sum(axis=1).flatten()

        Rq2 = safe_sparse_dot(Xq2_batch, VT.transpose()).dot(VT) - Xq2_batch
        nrq2 = np.sum(np.multiply(Rq2, Rq2), axis=1).flatten()

        q1res[start:finish] = np.sqrt(nrq1) / np.sqrt(nq1)
        q2res[start:finish] = np.sqrt(nrq2) / np.sqrt(nq2)
        eucl[start:finish] = euclidean(Rq1, Rq2).flatten()
        cos[start:finish] = cosine(Rq1, Rq2).flatten()

        start = finish

    train_df['svd_res_q1'] = q1res
    train_df['svd_res_q2'] = q2res
    train_df['svd_res_eucl'] = eucl
    train_df['svd_res_cos'] = cos

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        'svd_res_q1',
        'svd_res_q2',
        'svd_res_eucl',
        'svd_res_cos'
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['svddist.dataset'])

    logging.info('Computing test features')
    X = compute_feature_matrix(test_df, vectorizer, combine='stack')

    logging.info('Computing train SVD residuals')
    L = X.shape[0] / 2
    Xq1 = X[:L, :]
    Xq2 = X[L:, :]

    start = 0
    batch = 100
    eucl = np.zeros(Xq1.shape[0])
    cos = np.zeros(Xq1.shape[0])
    q1res = np.zeros(Xq1.shape[0])
    q2res = np.zeros(Xq1.shape[0])
    while start < Xq1.shape[0]:
        finish = min(start + batch, Xq1.shape[0])

        Xq1_batch = Xq1[start:finish, :]
        nq1 = (Xq1_batch.multiply(Xq1_batch)).sum(axis=1).flatten()

        Rq1 = safe_sparse_dot(Xq1_batch, VT.transpose()).dot(VT) - Xq1_batch
        nrq1 = np.sum(np.multiply(Rq1, Rq1), axis=1).flatten()

        Xq2_batch = Xq2[start:finish, :]
        nq2 = (Xq2_batch.multiply(Xq2_batch)).sum(axis=1).flatten()

        Rq2 = safe_sparse_dot(Xq2_batch, VT.transpose()).dot(VT) - Xq2_batch
        nrq2 = np.sum(np.multiply(Rq2, Rq2), axis=1).flatten()

        q1res[start:finish] = np.sqrt(nrq1) / np.sqrt(nq1)
        q2res[start:finish] = np.sqrt(nrq2) / np.sqrt(nq2)
        eucl[start:finish] = euclidean(Rq1, Rq2).flatten()
        cos[start:finish] = cosine(Rq1, Rq2).flatten()

        start = finish

    test_df['svd_res_q1'] = q1res
    test_df['svd_res_q2'] = q2res
    test_df['svd_res_eucl'] = eucl
    test_df['svd_res_cos'] = cos

    logging.info('Writing test dataset')
    test_df[[
        FieldsTest.test_id,
        'svd_res_q1',
        'svd_res_q2',
        'svd_res_eucl',
        'svd_res_cos'
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    main(project().conf)
