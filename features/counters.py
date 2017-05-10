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

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from lib.project import project
from lib.dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest


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
        auc_q1=auc_q1,
        auc_q2=auc_q2,
        correlation=correlation.to_json()
    )

    return quality, counts


def main(conf):
    logging.info('Loading train dataset')
    train_df = load_train_df()

    logging.info('Loading test dataset')
    test_df = load_test_df()

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
        FieldsTrain.freq_q2
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.freq_q1,
        FieldsTest.freq_q2
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
