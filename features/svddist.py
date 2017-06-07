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

from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.externals import joblib

from svd import train_vectorizer, compute_feature_matrix, compute_svd

import xgboost as xgb

from matplotlib import pyplot as plt


def plot_singular_values(s, img_dir):
    s = sorted(s, reverse=True)
    fh = plt.figure(1, figsize=[8, 4])
    plt.plot(s, 'b.-')
    plt.ylim([0, s[0]])
    plt.grid()
    plt.title('Singular values')
    fh.savefig(join_path(img_dir, 'singular_values.png'))


def plot_quality(quality, img_dir):
    fig1 = plt.figure(2, figsize=[8, 4])
    plt.plot(quality['roc']['train']['fpr'], quality['roc']['train']['tpr'], 'b-', label='train')
    plt.plot(quality['roc']['valid']['fpr'], quality['roc']['valid']['tpr'], 'r-', label='valid')
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k--')
    plt.legend()
    plt.grid()
    plt.title('ROC')
    fig1.savefig(join_path(img_dir, 'roc.png'))
    fig1.clf()

    fig2 = plt.figure(3, figsize=[8, 4])
    plt.plot(quality['reliability']['train']['avg_pred'], quality['reliability']['train']['avg_label'], 'b-', label='train')
    plt.plot(quality['reliability']['valid']['avg_pred'], quality['reliability']['valid']['avg_label'], 'r-', label='valid')
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k--')
    plt.legend()
    plt.grid()
    plt.title('Reliability')
    fig2.savefig(join_path(img_dir, 'reliability.png'))
    fig2.clf()


def train_xgb(X, y, skf, **options):
    quality = dict(folds=[], full=dict())
    predictions = np.zeros(len(y))

    dump_dir = options.get('dump_dir') or '.'

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        logging.info('Cross-validation fold: %d', i)
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        dump_file = join_path(dump_dir, 'model_%d.bin' % i)
        try:
            logging.info('Loading model for fold %d', i)
            f = xgb.Booster({'nthread': 4})
            f.load_model(dump_file)
        except:
            logging.info('Training model on fold %d', i)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            eval_list = [(dtrain, 'train'), (dvalid, 'valid')]

            f = xgb.train(options, dtrain, options['num_round'], eval_list)
            f.save_model(f, dump_file)

        p_train = f.predict(X_train)

        ll_train = log_loss(y_train, p_train)
        auc_train = roc_auc_score(y_train, p_train)

        logging.info('Train LL=%s AUC=%s', ll_train, auc_train)

        fpr_train, tpr_train, _ = roc_curve(y_train, p_train, pos_label=1)
        y_avg_train, p_avg_train = reliability_curve(y_train, p_train, nbins=50)

        p_valid = f.predict(X_valid)
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
                train=dict(fpr=fpr_train, tpr=tpr_train),
                valid=dict(fpr=fpr_valid, tpr=tpr_valid)
            ),
            reliability=dict(
                train=dict(avg_label=y_avg_train, avg_pred=p_avg_train),
                valid=dict(avg_label=y_avg_valid, avg_pred=p_avg_valid)
            )
        ))

    return quality, predictions


def main(conf):
    dump_dir = conf['svdxgb.dump.dir']
    makedirs(dump_dir)

    dump_config_file = join_path(dump_dir, 'application.conf')
    dump_config(conf, dump_config_file)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['svdxgb.dataset'])

    y = train_df['is_duplicate'].values

    vectorizer_file = join_path(dump_dir, 'vectorizer.pkl')
    try:
        logging.info('Loading vectorizer dump')
        vectorizer = joblib.load(vectorizer_file)
    except:
        logging.info('Loading vectorizer dump failed')
        logging.info('Traininig vectorizer')
        vectorizer = train_vectorizer(train_df, **conf['svdxgb.vectorizer'])

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
    k = conf['svdxgb.svd'].get_int('k')
    singular_values_file = join_path(dump_dir, 'singular_values.txt')
    singular_vectors_file = join_path(dump_dir, 'singular_vectors.npz')
    try:
        S = np.loadtxt(singular_values_file)
        VT = np.load(singular_vectors_file)['VT']
        assert k == len(S)
    except:
        logging.info('Loading SVD decomposition failed')
        logging.info('Computing SVD decomposition')
        S, VT = compute_svd(X.asfptype(), **conf['svdxgb.svd'])

        logging.info('Writing singular values to file')
        np.savetxt(singular_values_file, S)
        np.savez(singular_vectors_file, VT=VT)
        plot_singular_values(S, dump_dir)

    logging.info('Computing train SVD features')
    Sinv = np.diag(1. / S) * np.sqrt(X.shape[0])
    U = X.dot(VT.transpose().dot(Sinv))

    logging.info('Train feature matrix dimensions: %s', U.shape)

    logging.info('Symmetrizing input features')
    Uq1, Uq2 = np.vsplit(U, 2)
    U = np.hstack([(Uq1 + Uq2) / 2.0, (Uq1 - Uq2) / 2.0])

    logging.info('Training feature matrix: %s', U.shape)

    logging.info('Training feed-forward neural networks')
    quality, predictions = train_xgb(U, y, skfold(), dump_dir=dump_dir, **conf['svdxgb.model'])

    logging.info('Plotting quality metrics')
    quality_dir = join_path(dump_dir, 'quality')
    makedirs(quality_dir)
    for q in quality['folds']:
        img_dir = join_path(quality_dir, 'fold%d' % q['fold'])
        makedirs(img_dir)
        plot_quality(q, img_dir)

    logging.info('Writing train features')
    train_df['svdxgb'] = predictions

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        'svdxgb'
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['svdxgb.dataset'])

    logging.info('Computing test features')
    X = compute_feature_matrix(test_df, vectorizer, combine='stack')

    logging.info('Computing test SVD features')
    U = X.dot(VT.transpose().dot(Sinv))

    logging.info('Symmetrizing input features')
    Uq1, Uq2 = np.vsplit(U, 2)
    U = np.hstack([(Uq1 + Uq2) / 2.0, (Uq1 - Uq2) / 2.0])
    del Uq1, Uq2

    dtest = xgb.DMatrix(U)
    del U

    logging.info('Applying models to test dataset')
    test_df['svdxgb'] = np.zeros(U.shape[0])
    for q in quality['folds']:
        f = xgb.Booster({'nthread': 4})
        f.load_model(q['dump'])
        p = f.predict(dtest)
        test_df['svdxgb'] = test_df['svdxgb'] + logit(p)
    test_df['svdxgb'] = test_df['svdxgb'] / len(quality['folds'])

    logging.info('Writing test dataset')
    test_df[[
        FieldsTest.test_id,
        'svdxgb',
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
