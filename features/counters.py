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
            self.data[q1] = [next(self.sequence), 1]
        else:
            self.data[q1][1] += 1
        if q2 not in self.data:
            self.data[q2] = [next(self.sequence), 1]
        else:
            self.data[q2][1] += 1
        return self.data[q1][1], self.data[q2][1]


def counters(train_df, test_df, **options):

    # build collection of all unique questions
    questions = pd.concat([
        train_df[[FieldsTrain.question1]].rename(columns={'question1': 'q'}),
        train_df[[FieldsTrain.question2]].rename(columns={'question2': 'q'}),
        test_df[[FieldsTest.question1]].rename(columns={'question1': 'q'}),
        test_df[[FieldsTest.question2]].rename(columns={'question2': 'q'})
    ], ignore_index=True)

    counts = questions['q'].value_counts().to_dict()

    train_df[FieldsTrain.freq_q1] = train_df[FieldsTrain.question1].map(lambda q: np.log(counts.get(q, 1)))
    train_df[FieldsTrain.freq_q2] = train_df[FieldsTrain.question2].map(lambda q: np.log(counts.get(q, 1)))

    test_df[FieldsTest.freq_q1] = test_df[FieldsTest.question1].map(lambda q: np.log(counts.get(q, 1)))
    test_df[FieldsTest.freq_q2] = test_df[FieldsTest.question2].map(lambda q: np.log(counts.get(q, 1)))

    correlation = train_df[[FieldsTrain.is_duplicate, FieldsTrain.freq_q1, FieldsTrain.freq_q2]].corr()
    auc_q1 = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[FieldsTrain.freq_q1])
    auc_q2 = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[FieldsTrain.freq_q2])

    logging.info("Frequency of question1 AUC=%s", auc_q1)
    logging.info("Frequency of question2 AUC=%s", auc_q2)

    quality = dict(
        auc_freq_q1=auc_q1,
        auc_freq_q2=auc_q2,
        correlation_freq=correlation.to_json()
    )

    hashing = HashCounter()
    train_df['count_q1'], train_df['count_q2'] = zip(*train_df.apply(lambda r: hashing.update(r[Fields.question1], r[Fields.question2]), axis=1))
    test_df['count_q1'], test_df['count_q2'] = zip(*test_df.apply(lambda r: hashing.update(r[Fields.question1], r[Fields.question2]), axis=1))

    correlation = train_df[[FieldsTrain.is_duplicate, FieldsTrain.count_q1, FieldsTrain.count_q2]].corr()
    auc_q1 = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[FieldsTrain.count_q1])
    auc_q2 = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[FieldsTrain.count_q2])

    logging.info("Count of question1 AUC=%s", auc_q1)
    logging.info("Count of question2 AUC=%s", auc_q2)
    print correlation
    quality.update(dict(
        auc_count_q1=auc_q1,
        auc_count_q2=auc_q2
    ))

    return quality, counts


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df(conf['counters.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['counters.dataset'])

    logging.info('Computing question frequencies')
    quality, counts = counters(train_df, test_df)

    logging.info('Writing dump')
    dump_dir = conf['counters.dump.dir']

    try:
        makedirs(dump_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(join_path(dump_dir, 'quality.json'), 'w') as quality_file:
        json.dump(quality, quality_file)

    counts_file = join_path(dump_dir, 'counts.pkl')
    joblib.dump(counts, counts_file)

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.freq_q1,
        FieldsTrain.freq_q2,
        FieldsTrain.count_q1,
        FieldsTrain.count_q2
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.freq_q1,
        FieldsTest.freq_q2,
        FieldsTest.count_q1,
        FieldsTest.count_q2
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
