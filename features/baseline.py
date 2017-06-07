# -*- coding: utf-8 -*-

"""
Abhishek features + leaky + simple
https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky
"""

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


def word_match_share(row, stops=None):
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
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    r = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return r


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return len(wic) / len(uw)


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1'])*1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


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


def compute_features(train_df, test_df):
    train_df['question1'] = train_df['question1'].map(lambda x: str(x).lower().split())
    train_df['question2'] = train_df['question2'].map(lambda x: str(x).lower().split())

    test_df['question1'] = test_df['question1'].map(lambda x: str(x).lower().split())
    test_df['question2'] = test_df['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    f = functools.partial(word_match_share, stops=stops)
    train_df[FieldsTrain.word_match] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.word_match] = test_df.apply(f, axis=1, raw=True)

    train_df[FieldsTrain.jaccard] = train_df.apply(jaccard, axis=1, raw=True)
    test_df[FieldsTest.jaccard] = test_df.apply(jaccard, axis=1, raw=True)

    train_df[FieldsTrain.wc_diff] = train_df.apply(wc_diff, axis=1, raw=True)
    test_df[FieldsTest.wc_diff] = test_df.apply(wc_diff, axis=1, raw=True)

    train_df[FieldsTrain.wc_ratio] = train_df.apply(wc_ratio, axis=1, raw=True)
    test_df[FieldsTest.wc_ratio] = test_df.apply(wc_ratio, axis=1, raw=True)

    train_df[FieldsTrain.wc_diff_unique] = train_df.apply(wc_diff_unique, axis=1, raw=True)
    test_df[FieldsTest.wc_diff_unique] = test_df.apply(wc_diff_unique, axis=1, raw=True)

    train_df[FieldsTrain.wc_ratio_unique] = train_df.apply(wc_ratio_unique, axis=1, raw=True)
    test_df[FieldsTest.wc_ratio_unique] = test_df.apply(wc_ratio_unique, axis=1, raw=True)

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    train_df[FieldsTrain.wc_diff_unq_stop] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.wc_diff_unq_stop] = test_df.apply(f, axis=1, raw=True)

    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    train_df[FieldsTrain.wc_ratio_unique_stop] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.wc_ratio_unique_stop] = test_df.apply(f, axis=1, raw=True)

    train_df[FieldsTrain.same_start] = train_df.apply(same_start_word, axis=1, raw=True)
    test_df[FieldsTest.same_start] = test_df.apply(same_start_word, axis=1, raw=True)

    train_df[FieldsTrain.char_diff] = train_df.apply(char_diff, axis=1, raw=True)
    test_df[FieldsTest.char_diff] = test_df.apply(char_diff, axis=1, raw=True)

    f = functools.partial(char_diff_unique_stop, stops=stops)
    train_df[FieldsTrain.char_diff_unq_stop] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.char_diff_unq_stop] = test_df.apply(f, axis=1, raw=True)

    train_df[FieldsTrain.total_unique_words] = train_df.apply(total_unique_words, axis=1, raw=True)
    test_df[FieldsTest.total_unique_words] = test_df.apply(total_unique_words, axis=1, raw=True)

    f = functools.partial(total_unq_words_stop, stops=stops)
    train_df[FieldsTrain.total_unq_words_stop] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.total_unq_words_stop] = test_df.apply(f, axis=1, raw=True)

    train_df[FieldsTrain.char_ratio] = train_df.apply(char_ratio, axis=1, raw=True)
    test_df[FieldsTest.char_ratio] = test_df.apply(char_ratio, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share, weights=weights)
    train_df[FieldsTrain.tfidf_wm] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.tfidf_wm] = test_df.apply(f, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    train_df[FieldsTrain.tfidf_wm_stops] = train_df.apply(f, axis=1, raw=True)
    test_df[FieldsTest.tfidf_wm_stops] = test_df.apply(f, axis=1, raw=True)


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['baseline.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['baseline.dataset'])

    logging.info('Computing baseline features')
    compute_features(train_df, test_df)

    logging.info('Writing dump')
    dump_dir = conf['baseline.dump.dir']
    makedirs(dump_dir)

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.word_match,
        FieldsTrain.jaccard,
        FieldsTrain.wc_diff,
        FieldsTrain.wc_ratio,
        FieldsTrain.wc_diff_unique,
        FieldsTrain.wc_ratio_unique,
        FieldsTrain.wc_diff_unq_stop,
        FieldsTrain.wc_ratio_unique_stop,
        FieldsTrain.same_start,
        FieldsTrain.char_diff,
        FieldsTrain.char_diff_unq_stop,
        FieldsTrain.total_unique_words,
        FieldsTrain.total_unq_words_stop,
        FieldsTrain.char_ratio,
        FieldsTrain.tfidf_wm,
        FieldsTrain.tfidf_wm_stops
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.word_match,
        FieldsTest.jaccard,
        FieldsTest.wc_diff,
        FieldsTest.wc_ratio,
        FieldsTest.wc_diff_unique,
        FieldsTest.wc_ratio_unique,
        FieldsTest.wc_diff_unq_stop,
        FieldsTest.wc_ratio_unique_stop,
        FieldsTest.same_start,
        FieldsTest.char_diff,
        FieldsTest.char_diff_unq_stop,
        FieldsTest.total_unique_words,
        FieldsTest.total_unq_words_stop,
        FieldsTest.char_ratio,
        FieldsTest.tfidf_wm,
        FieldsTest.tfidf_wm_stops
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)

