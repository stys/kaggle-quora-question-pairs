# -*- coding: utf-8 -*-

from __future__ import division

import logging
import functools
from os.path import join as join_path
from collections import Counter

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

from lib.project import project
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest
from lib.utils import makedirs


def tfidf_word_match_share(row, weights=None):

    q1words = {}
    for word in row['question1']:
        q1words[word] = 1

    q2words = {}
    for word in row['question2']:
        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = np.sum([weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words])

    total_weights = np.sum([weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words])

    if total_weights > 0:
        return shared_weights / total_weights
    else:
        return 0


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = np.sum([weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words])
    total_weights = np.sum([weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words])

    if total_weights > 0:
        return shared_weights / total_weights
    else:
        return 0


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def compute_tfidf_features(train_df, test_df):
    train_df['question1'] = train_df['question1'].map(lambda x: str(x).lower().split())
    train_df['question2'] = train_df['question2'].map(lambda x: str(x).lower().split())

    test_df['question1'] = test_df['question1'].map(lambda x: str(x).lower().split())
    test_df['question2'] = test_df['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    f = functools.partial(tfidf_word_match_share, weights=weights)
    train_df[FieldsTrain.tfidf_wm] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.tfidf_wm] = test_df.apply(f, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    train_df[FieldsTrain.tfidf_wm_stops] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.tfidf_wm_stops] = test_df.apply(f, axis=1, raw=True)


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['tfidf.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['tfidf.dataset'])

    logging.info('Computing tfidf features')
    compute_tfidf_features(train_df, test_df)

    logging.info('Writing dump')
    dump_dir = conf['tfidf.dump.dir']
    makedirs(dump_dir)

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.tfidf_wm,
        FieldsTrain.tfidf_wm_stops
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.tfidf_wm,
        FieldsTest.tfidf_wm_stops
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)

