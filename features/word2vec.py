# -*- coding: utf-8 -*-

"""
Word2vec features from Abhishek's features
https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py
"""

import logging
from os.path import join as join_path

from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest
from lib.project import project
from lib.utils import makedirs

import numpy as np

from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis

from sklearn.metrics import roc_auc_score

import gensim
from nltk.corpus import stopwords as nltk_stopwords

from tqdm import tqdm
tqdm.pandas(desc='progress')

stopwords = nltk_stopwords.words('english')


class Word2Vec(object):
    def __init__(self, model, model_norm):
        self.model = model
        self.model_norm = model_norm

    def features(self, q1, q2):
        q1 = str(q1).lower().split()
        q2 = str(q2).lower().split()
        q1 = [w for w in q1 if w not in stopwords]
        q2 = [w for w in q2 if w not in stopwords]

        wmd = min(self.model.wmdistance(q1, q2), 10)
        wmd_norm = min(self.model_norm.wmdistance(q1, q2), 10)

        q1vec = self.sent2vec(q1)
        q2vec = self.sent2vec(q2)

        if q1vec is not None and q2vec is not None:
            cos = cosine(q1vec, q2vec)
            city = cityblock(q1vec, q2vec)
            jacc = jaccard(q1vec, q2vec)
            canb = canberra(q1vec, q2vec)
            eucl = euclidean(q1vec, q2vec)
            mink = minkowski(q1vec, q2vec, 3)
            bray = braycurtis(q1vec, q2vec)

            q1_skew = skew(q1vec)
            q2_skew = skew(q2vec)
            q1_kurt = kurtosis(q1vec)
            q2_kurt = kurtosis(q2vec)

        else:
            cos = -1
            city = -1
            jacc = -1
            canb = -1
            eucl = -1
            mink = -1
            bray = -1

            q1_skew = 0
            q2_skew = 0
            q1_kurt = 0
            q2_kurt = 0

        return wmd, wmd_norm, cos, city, jacc, canb, eucl, mink, bray, q1_skew, q2_skew, q1_kurt, q2_kurt

    def sent2vec(self, words):
        M = []
        for w in words:
            try:
                M.append(self.model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        norm = np.sqrt((v ** 2).sum())
        if norm > 0:
            return v / np.sqrt((v ** 2).sum())
        else:
            return None


def main(conf):
    dump_dir = conf['word2vec']['dump']['dir']
    makedirs(dump_dir)

    logging.warning('Loading train dataset')
    train_df = load_train_df(conf['word2vec']['dataset'])

    logging.warning('Loading test dataset')
    test_df = load_test_df(conf['word2vec']['dataset'])

    logging.warning('Loading embeddings')
    embeddings_dir = conf['word2vec']['embeddings']['dir']
    embeddings_file = join_path(embeddings_dir, conf['word2vec']['embeddings']['file'])
    w2v = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
    w2v_norm = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
    w2v_norm.init_sims(replace=True)
    processor = Word2Vec(w2v, w2v_norm)

    logging.warning('Computing train features')

    train_df[Fields.w2v_wmd], \
    train_df[Fields.w2v_wmd_norm], \
    train_df[Fields.w2v_cos], \
    train_df[Fields.w2v_city], \
    train_df[Fields.w2v_jacc], \
    train_df[Fields.w2v_canb], \
    train_df[Fields.w2v_eucl], \
    train_df[Fields.w2v_mink], \
    train_df[Fields.w2v_bray], \
    train_df[Fields.w2v_skew_q1], \
    train_df[Fields.w2v_skew_q2], \
    train_df[Fields.w2v_kurt_q1], \
    train_df[Fields.w2v_kurt_q2] = \
        zip(*train_df.progress_apply(lambda r: processor.features(r['question1'], r['question2']), axis=1))

    for feature in [f for f in dir(Fields()) if f.startswith('w2v')]:
        logging.warning('Feature %s AUC=%s', feature, roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature]))

    logging.warning('Writing train feature dump')
    train_df.drop([Fields.question1, Fields.question2, FieldsTrain.qid1, FieldsTrain.qid2], axis=1, inplace=True)
    train_df.to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.warning('Computing test features')
    test_df[Fields.w2v_wmd], \
    test_df[Fields.w2v_wmd_norm], \
    test_df[Fields.w2v_cos], \
    test_df[Fields.w2v_city], \
    test_df[Fields.w2v_jacc], \
    test_df[Fields.w2v_canb], \
    test_df[Fields.w2v_eucl], \
    test_df[Fields.w2v_mink], \
    test_df[Fields.w2v_bray], \
    test_df[Fields.w2v_skew_q1], \
    test_df[Fields.w2v_skew_q2], \
    test_df[Fields.w2v_kurt_q1], \
    test_df[Fields.w2v_kurt_q2] = \
        zip(*test_df.progress_apply(lambda r: processor.features(r['question1'], r['question2']), axis=1))

    logging.warning('Writing test feature dump')
    test_df.drop([Fields.question1, Fields.question2], axis=1, inplace=True)
    test_df.to_csv(join_path(dump_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    main(project().conf)

