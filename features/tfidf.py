import errno
import logging
from os import makedirs
from os.path import join as join_path

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from lib.project import project

from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest
from lib.utils import makedirs
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

from scipy import sparse

from tqdm import tqdm_notebook


def compute_quality(train_df, feature):
    corr = train_df[[FieldsTrain.is_duplicate, feature]].corr().values.tolist()
    auc = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)


def compute_m_svd(m_q1_tf, m_q2_tf):
    logging.info('Computing SVD')
    svd = TruncatedSVD(n_components=100)
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
        data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
        data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

    data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

    return data


def getTrigramsIndex(cv_char):
    trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
    return np.sort(trigrams.values())


def compute_features(train_df, test_df):
    cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
    cv_char.fit_transform(train_df[FieldsTrain.question1].tolist() + train_df[FieldsTrain.question2].tolist())

    logging.info('Trigrams')
    ix_trigrams_train = getTrigramsIndex(cv_char)
    ix_trigrams_test = getTrigramsIndex(cv_char)

    tft = TfidfTransformer(
        norm='l2',
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False)

    # Train
    logging.info('Tf-idf transformation for training data')
    m_q1 = cv_char.transform(train_df[FieldsTrain.question1].values)[:, ix_trigrams_train]
    m_q2 = cv_char.transform(train_df[FieldsTrain.question2].values)[:, ix_trigrams_train]
    tft = tft.fit(sparse.vstack((m_q1, m_q2)))
    m_q1_tf_train = tft.transform(m_q1)
    m_q2_tf_train = tft.transform(m_q2)

    # Test
    logging.info('Tf-idf transformation for test data')
    m_q1 = cv_char.transform(test_df[FieldsTest.question1].values)[:, ix_trigrams_test]
    m_q2 = cv_char.transform(test_df[FieldsTest.question2].values)[:, ix_trigrams_test]
    tft = tft.fit(sparse.vstack((m_q1, m_q2)))
    m_q1_tf_test = tft.transform(m_q1)
    m_q2_tf_test = tft.transform(m_q2)

    train_df[FieldsTrain.m_q1_q2_tf_svd0] = compute_m_svd(m_q1_tf_train, m_q2_tf_train)[:,0]
    test_df[FieldsTest.m_q1_q2_tf_svd0] = compute_m_svd(m_q1_tf_test, m_q2_tf_test)[:,0]
    quality_m_q1_q2_tf_svd0 = compute_quality(train_df, Fields.m_q1_q2_tf_svd0)

    data = {
        'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf_train, m_q2_tf_train))),
        'y_train': train_df[FieldsTrain.is_duplicate],
        'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf_test, m_q2_tf_test))),
        'y_train_pred': np.zeros(train_df.shape[0]),
        'y_test_pred': []
    }

    data = compute_sgd(data)

    train_df[FieldsTrain.m_w1l_tfidf_oof] = np.zeros(train_df.shape[0])
    train_df.loc[FieldsTrain.m_w1l_tfidf_oof] = data['y_train_pred']

    test_df[FieldsTest.m_w1l_tfidf_oof] = np.zeros(test_df.shape[0])
    test_df.loc[FieldsTest.m_w1l_tfidf_oof] = data['y_test_pred_fixed']
    quality_m_w1l_tfidf_oof = compute_quality(train_df, Fields.m_w1l_tfidf_oof)


    quality = dict(
        quality_m_q1_q2_tf_svd0=quality_m_q1_q2_tf_svd0,
        quality_m_w1l_tfidf_oof=quality_m_w1l_tfidf_oof
    )

    return quality


def main(conf):
    dump_dir = conf['tfidf.dump.dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['tfidf.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['tfidf.dataset'])

    compute_features(train_df, test_df)

    logging.info('Writing train dataset to disk')
    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        Fields.m_q1_q2_tf_svd0
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Writing test dataset to disk')
    test_df[[
        FieldsTest.test_id,
        Fields.m_q1_q2_tf_svd0
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
