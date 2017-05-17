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
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest, skfold
from lib.quality import reliability_curve

from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.externals import joblib

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Activation

from svd import train_vectorizer, compute_feature_matrix, compute_svd

from matplotlib import pyplot as plt


def plot_singular_values(s, img_dir):
    s = sorted(s, reverse=True)
    fh = plt.figure(1, figsize=[8, 4])
    plt.plot(s, 'b.-')
    plt.ylim([0, s[0]])
    plt.grid()
    plt.title('Singular values')
    fh.savefig(join_path(img_dir, 'singular_values.png'))


def train_ff(X, y, skf, **options):
    quality = dict(folds=[], full=dict())
    predictions = np.zeros(len(y))

    hidden_layers = options.get('hidden_layers') or (50, 10)
    activation = options.get('activation') or 'relu'
    alpha = options.get('alpha') or 0.01
    max_iter = options.get('max_iter') or 200
    tol = options.get('tol') or 0.0
    dump_dir = options.get('dump_dir') or '.'

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]

        dump_file = join_path(dump_dir, 'model_%d.pkl' % i)
        try:
            logging.info('Loading model for fold %d', i)
            f = load_model(dump_file)
        except:
            logging.info('Training model on fold %d', i)
            f = Sequential()
            f.add(Dense(10, input_dim=40))
            f.add(Activation('relu'))
            f.add(Dense(1))
            f.add(Activation('sigmoid'))
            f.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            f.fit(X_train, y_train, epochs=5, batch_size=32)

        p_train = f.predict_proba(X_train)
        ll_train = log_loss(y_train, p_train)
        auc_train = roc_auc_score(y_train, p_train)

        logging.info('Train LL=%s AUC=%s', ll_train, auc_train)

        fpr_train, tpr_train, _ = roc_curve(y_train, p_train, pos_label=1)
        y_avg_train, p_avg_train = reliability_curve(y_train, p_train, nbins=50)

        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        p_valid = f.predict_proba(X_valid)
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

    return quality, predictions


def main(conf):
    dump_dir = conf['svdff.dump.dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['svdff.dataset'])

    vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
    try:
        logging.info('Loading vectorizer dump')
        vectorizer = joblib.load(vectorizer_file)
    except:
        logging.info('Loading vectorizer dump failed')
        logging.info('Traininig vectorizer')
        vectorizer = train_vectorizer(train_df, **conf['svdff.vectorizer'])

        logging.info('Writing vectorizer dump')
        joblib.dump(vectorizer, vectorizer_file)

    logging.info('Computing train sparse feature matrix')
    X = compute_feature_matrix(train_df, vectorizer, combine=conf.get('svdff.svd.transform', None))

    logging.info('Loading SVD decomposition')
    k = conf['svdff.svd'].get_int('k')
    singular_values_file = join_path(dump_dir, 'singular_values.txt')
    singular_vectors_file = join_path(dump_dir, 'singular_vectors.npz')
    try:
        S = np.loadtxt(singular_values_file)
        VT = np.load(singular_vectors_file)
        assert k == len(S)
    except:
        logging.info('Loading SVD decomposition failed')
        logging.info('Computing SVD decomposition')
        S, VT = compute_svd(X.asfptype(), **conf['svdff.svd'])

        logging.info('Writing singular values to file')
        np.savetxt(singular_values_file, S)
        np.savez(singular_vectors_file, VT)
        plot_singular_values(S, dump_dir)

    plot_singular_values(S, dump_dir)

    # logging.info('Computing train SVD features')
    # Sinv = np.diag(1. / S) * np.sqrt(X.shape[0])
    # U = X.dot(VT.transpose()).dot(Sinv)

if __name__ == '__main__':
    main(project().conf)
