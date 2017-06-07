import errno
import logging
from os import makedirs
from os.path import join as join_path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from lib.project import project

from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest
from lib.utils import makedirs

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

from scipy import sparse
from scipy.special import logit

from tqdm import tqdm_notebook


def compute_quality(train_df, feature):
    corr = train_df[[FieldsTrain.is_duplicate, feature]].corr().values.tolist()
    auc = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)


def compute_m_svd(m_q1_tf, m_q2_tf):
    logging.info('Computing SVD')
    svd = TruncatedSVD(n_components=3)
    m_svd = svd.fit_transform(sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf))))
    return m_svd


def compute_sgd(data):
    logging.info('Computing SGD')

    n_splits = 10
    folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']),
                                             total=n_splits):
        # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
        model = SGDClassifier(
            loss='log',
            penalty='elasticnet',
            fit_intercept=True,
            n_iter=100,
            shuffle=True,
            n_jobs=-1,
            l1_ratio=0.0001,
            alpha=1e-05,
            class_weight=None)
        model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
        data['y_train_pred'][ix_second] = logit(model.predict_proba(data['X_train'][ix_second, :])[:, 1])
        data['y_test_pred'].append(logit(model.predict_proba(data['X_test'])[:, 1]))

    data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

    return data


def getTrigramsIndex(cv_char):
    trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
    return np.sort(trigrams.values())


def compute_features(train_df, test_df):
    logging.info('Preparing dataset')
    train_df['test_id'] = -1
    test_df['id'] = -1
    test_df['qid1'] = -1
    test_df['qid2'] = -1
    test_df['is_duplicate'] = -1

    df = pd.concat([train_df, test_df])
    df['uid'] = np.arange(df.shape[0])
    df = df.set_index(['uid'])

    ix_train = np.where(df['id'] >= 0)[0]
    ix_test = np.where(df['id'] == -1)[0]
    ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
    ix_not_dup = np.where(df['is_duplicate'] == 0)[0]

    logging.info('Building count vectroizer')
    cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char', min_df=5)
    cv_char.fit_transform(df[Fields.question1].tolist() + df[Fields.question2].tolist())

    logging.info('Building unigram index')
    unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
    ix_unigrams = np.sort(unigrams.values())

    logging.info('Building bigram index')
    bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
    ix_bigrams = np.sort(bigrams.values())

    logging.info('Building trigram index')
    trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
    ix_trigrams = np.sort(trigrams.values())

    m_q1 = cv_char.transform(df['question1'].values)
    m_q2 = cv_char.transform(df['question2'].values)

    logging.info('Computing unigram features')
    v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['unigram_all_jaccard'] = v_score[ix_train]
    test_df['unigram_all_jaccard'] = v_score[ix_test]

    v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['unigram_all_jaccard_max'] = v_score[ix_train]
    test_df['unigram_all_jaccard_max'] = v_score[ix_test]

    logging.info('Computing bigram features')
    v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['bigram_all_jaccard'] = v_score[ix_train]
    test_df['bigram_all_jaccard'] = v_score[ix_test]

    v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['bigram_all_jaccard_max'] = v_score[ix_train]
    test_df['bigram_all_jaccard_max'] = v_score[ix_test]

    logging.info('Computing trigram features')
    m_q1 = m_q1[:, ix_trigrams]
    m_q2 = m_q2[:, ix_trigrams]

    v_num = m_q1.minimum(m_q2).sum(axis=1)
    v_den = m_q1.sum(axis=1) + m_q2.sum(axis=1)
    v_den[np.where(v_den == 0)] = 1
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['trigram_all_jaccard'] = v_score[ix_train]
    test_df['trigram_all_jaccard'] = v_score[ix_test]

    v_num = m_q1.minimum(m_q2).sum(axis=1)
    v_den = m_q1.maximum(m_q2).sum(axis=1)
    v_den[np.where(v_den == 0)] = 1
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]

    train_df['trigram_all_jaccard_max'] = v_score[ix_train]
    test_df['trigram_all_jaccard_max'] = v_score[ix_test]

    logging.info('Computing trrigram distance features')
    tft = TfidfTransformer(
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False)

    tft = tft.fit(sparse.vstack((m_q1, m_q2)))
    m_q1_tf = tft.transform(m_q1)
    m_q2_tf = tft.transform(m_q2)

    v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
    v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
            np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
    v_num[np.where(v_den == 0)] = 1
    v_den[np.where(v_den == 0)] = 1

    v_score = 1 - v_num / v_den

    train_df['trigram_tfidf_cosine'] = v_score[ix_train]
    test_df['trigram_tfidf_cosine'] = v_score[ix_test]

    tft = TfidfTransformer(
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False)

    tft = tft.fit(sparse.vstack((m_q1, m_q2)))
    m_q1_tf = tft.transform(m_q1)
    m_q2_tf = tft.transform(m_q2)

    v_score = (m_q1_tf - m_q2_tf)
    v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

    train_df['trigram_tfidf_l2_euclidean'] = v_score[ix_train]
    test_df['trigram_tfidf_l2_euclidean'] = v_score[ix_test]

    logging.info('Computing SVD features')
    tft = TfidfTransformer(
        norm='l2',
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False)

    tft = tft.fit(sparse.vstack((m_q1, m_q2)))
    m_q1_tf = tft.transform(m_q1)
    m_q2_tf = tft.transform(m_q2)

    svd = compute_m_svd(m_q1_tf, m_q2_tf)
    train_df['m_q1_q2_tf_svd0'] = svd[ix_train, 0]
    test_df['m_q1_q2_tf_svd0'] = svd[ix_test, 0]
    train_df['m_q1_q2_tf_svd1'] = svd[ix_train, 1]
    test_df['m_q1_q2_tf_svd1'] = svd[ix_test, 1]
    train_df['m_q1_q2_tf_svd2'] = svd[ix_train, 2]
    test_df['m_q1_q2_tf_svd2'] = svd[ix_test, 2]

    # logging.info('Building tfidf linear models')
    # data = {
    #     'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    #     'y_train': df.loc[ix_train]['is_duplicate'],
    #     'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    #     'y_train_pred': np.zeros(train_df.shape[0]),
    #     'y_test_pred': []
    # }
    #
    # data = compute_sgd(data)
    #
    # train_df[FieldsTrain.m_w1l_tfidf_oof] = np.zeros(train_df.shape[0])
    # train_df.loc[FieldsTrain.m_w1l_tfidf_oof] = data['y_train_pred']
    #
    # test_df[FieldsTest.m_w1l_tfidf_oof] = np.zeros(test_df.shape[0])
    # test_df.loc[FieldsTest.m_w1l_tfidf_oof] = data['y_test_pred']


def main(conf):
    dump_dir = conf['mephistopheies.dump.dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['mephistopheies.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['mephistopheies.dataset'])

    compute_features(train_df, test_df)

    logging.info('Writing train dataset to disk')
    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.unigram_all_jaccard,
        FieldsTrain.unigram_all_jaccard_max,
        FieldsTrain.bigram_all_jaccard,
        FieldsTrain.bigram_all_jaccard_max,
        FieldsTrain.trigram_all_jaccard,
        FieldsTrain.trigram_all_jaccard_max,
        FieldsTrain.trigram_tfidf_cosine,
        FieldsTrain.trigram_tfidf_l2_euclidean,
        FieldsTrain.m_q1_q2_tf_svd0,
        FieldsTrain.m_q1_q2_tf_svd1,
        FieldsTrain.m_q1_q2_tf_svd2,
        #FieldsTrain.m_w1l_tfidf_oof
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Writing test dataset to disk')
    test_df[[
        FieldsTest.test_id,
        FieldsTest.unigram_all_jaccard,
        FieldsTest.unigram_all_jaccard_max,
        FieldsTest.bigram_all_jaccard,
        FieldsTest.bigram_all_jaccard_max,
        FieldsTest.trigram_all_jaccard,
        FieldsTest.trigram_all_jaccard_max,
        FieldsTest.trigram_tfidf_cosine,
        FieldsTest.trigram_tfidf_l2_euclidean,
        FieldsTest.m_q1_q2_tf_svd0,
        FieldsTest.m_q1_q2_tf_svd1,
        FieldsTest.m_q1_q2_tf_svd2,
        #FieldsTest.m_w1l_tfidf_oof
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)