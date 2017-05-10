# -*- coding: utf-8 -*-

"""
Simplest features from Abhishek's fs-1
https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
"""

import errno
import logging
import re
from os import makedirs
from os.path import join as join_path

from sklearn.metrics import roc_auc_score

from lib.project import project
from lib.dataset import Fields, FieldsTrain, FieldsTest, load_train_df, load_test_df


def compute_quality(train_df, feature):
    corr = train_df[[FieldsTrain.is_duplicate, feature]].corr().values.tolist()
    auc = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)


def compute_features(train_df, test_df):

    train_df[Fields.len_q1] = train_df[Fields.question1].map(lambda q: len(q))
    test_df[Fields.len_q1] = test_df[Fields.question1].map(lambda q: len(q))
    quality_len_q1 = compute_quality(train_df, Fields.len_q1)

    train_df[Fields.len_q2] = train_df[Fields.question2].map(lambda q: len(q))
    test_df[Fields.len_q2] = test_df[Fields.question2].map(lambda q: len(q))
    quality_len_q2 = compute_quality(train_df, FieldsTrain.len_q2)

    train_df[Fields.diff_len] = abs(train_df[Fields.len_q1] - train_df[Fields.len_q2])
    test_df[Fields.diff_len] = abs(test_df[Fields.len_q1] - test_df[Fields.len_q2])
    quality_diff_len = compute_quality(train_df, FieldsTrain.diff_len)

    train_df[Fields.len_word_q1] = train_df[Fields.question1].map(lambda q: len(q.split()))
    test_df[Fields.len_word_q1] = test_df[Fields.question1].map(lambda q: len(q.split()))
    quality_len_word_q1 = compute_quality(train_df, Fields.len_word_q1)

    train_df[Fields.len_word_q2] = train_df[Fields.question2].map(lambda q: len(q.split()))
    test_df[Fields.len_word_q2] = test_df[Fields.question2].map(lambda q: len(q.split()))
    quality_len_word_q2 = compute_quality(train_df, Fields.len_word_q2)

    train_df[Fields.diff_len_word] = abs(train_df[Fields.len_word_q1] - train_df[Fields.len_word_q2])
    test_df[Fields.diff_len_word] = abs(test_df[Fields.len_word_q1] - test_df[Fields.len_word_q2])
    quality_diff_len_word = compute_quality(train_df, Fields.diff_len_word)

    pattern = re.compile('\s.')
    train_df[Fields.len_char_q1] = train_df[Fields.question1].map(lambda q: len(re.sub(pattern, '', q)))
    test_df[Fields.len_char_q1] = test_df[Fields.question1].map(lambda q: len(re.sub(pattern, '', q)))
    quality_len_char_q1 = compute_quality(train_df, Fields.len_char_q1)

    train_df[Fields.len_char_q2] = train_df[Fields.question2].map(lambda q: len(re.sub(pattern, '', q)))
    test_df[Fields.len_char_q2] = test_df[Fields.question2].map(lambda q: len(re.sub(pattern, '', q)))
    quality_len_char_q2 = compute_quality(train_df, Fields.len_char_q2)

    train_df[Fields.diff_len_char] = abs(train_df[Fields.len_char_q1] - train_df[Fields.len_char_q2])
    test_df[Fields.diff_len_char] = abs(test_df[Fields.len_char_q1] - test_df[Fields.len_char_q2])
    quality_diff_len_char = compute_quality(train_df, Fields.diff_len_char)

    quality=dict(
        quality_len_q1=quality_len_q1,
        quality_len_q2=quality_len_q2,
        quality_diff_len=quality_diff_len,
        quality_len_word_q1=quality_len_word_q1,
        quality_len_word_q2=quality_len_word_q2,
        quality_diff_len_word=quality_diff_len_word,
        quality_len_char_q1=quality_len_char_q1,
        quality_len_char_q2=quality_len_char_q2,
        quality_diff_len_char=quality_diff_len_char
    )

    return quality


def main(conf):
    dump_dir = conf['simplest.dump.dir']

    try:
        makedirs(dump_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    logging.info('Loading train dataset')
    train_df = load_train_df()

    logging.info('Loading test dataset')
    test_df = load_test_df()

    compute_features(train_df, test_df)

    logging.info('Writing train dataset to disk')
    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        Fields.len_q1,
        Fields.len_q2,
        Fields.diff_len,
        Fields.len_word_q1,
        Fields.len_word_q2,
        Fields.diff_len_word,
        Fields.len_char_q1,
        Fields.len_char_q2,
        Fields.diff_len_char
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Writing test dataset to disk')
    test_df[[
        FieldsTest.test_id,
        Fields.len_q1,
        Fields.len_q2,
        Fields.diff_len,
        Fields.len_word_q1,
        Fields.len_word_q2,
        Fields.diff_len_word,
        Fields.len_char_q1,
        Fields.len_char_q2,
        Fields.diff_len_char
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
