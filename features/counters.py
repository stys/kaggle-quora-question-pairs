# -*- coding: utf-8 -*-

"""
Question frequencies are correlated with duplicate probability
https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
"""

import errno
import json
import logging
from os import makedirs
from os.path import join as join_path
from itertools import count
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from lib.project import project
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest


class HashCounter(object):
    def __init__(self):
        self.data = dict()
        self.sequence = count()

    def update(self, q1, q2):
        if q1 not in self.data:
            self.data[q1] = next(self.sequence)

        if q2 not in self.data:
            self.data[q2] = next(self.sequence)

        return self.data[q1], self.data[q2]


def compute_counters(train_df, test_df, **options):
    ques = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return (len(q_dict[row['question1']]))

    def q2_freq(row):
        return (len(q_dict[row['question2']]))

    def q1_q2_intersect(row):
        return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    def q1_q2_intersect_second_order(row):
        q1 = row['question1']
        q2 = row['question2']

        q1_neighbours = set(q_dict[q1])
        q1_neighbours_second_order = set(k for q in q1_neighbours for k in set(q_dict[q]) if k != q1 and k != q2)

        q2_neighbours = set(q_dict[q2])
        q2_neighbours_second_order = set(k for q in q2_neighbours for k in set(q_dict[q]) if k != q1 and k != q2)

        return len(q1_neighbours_second_order.intersection(q2_neighbours_second_order))

    train_df[Fields.intersect_q1_q2] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
    train_df[Fields.intersect2_q1_q2] = train_df.apply(q1_q2_intersect_second_order, axis=1, raw=True)
    train_df[Fields.freq_q1] = train_df.apply(q1_freq, axis=1, raw=True)
    train_df[Fields.freq_q2] = train_df.apply(q2_freq, axis=1, raw=True)

    test_df[Fields.intersect_q1_q2] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
    test_df[Fields.intersect2_q1_q2] = test_df.apply(q1_q2_intersect_second_order, axis=1, raw=True)
    test_df[Fields.freq_q1] = test_df.apply(q1_freq, axis=1, raw=True)
    test_df[Fields.freq_q2] = test_df.apply(q2_freq, axis=1, raw=True)


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['counters.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['counters.dataset'])

    logging.info('Computing question frequencies')
    compute_counters(train_df, test_df)

    logging.info('Writing dump')
    dump_dir = conf['counters.dump.dir']

    try:
        makedirs(dump_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.freq_q1,
        FieldsTrain.freq_q2,
        FieldsTrain.intersect_q1_q2,
        FieldsTrain.intersect2_q1_q2
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.freq_q1,
        FieldsTest.freq_q2,
        FieldsTest.intersect_q1_q2,
        FieldsTest.intersect2_q1_q2
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
