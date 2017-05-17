# -*- coding: utf-8 -*-

"""
From
https://www.kaggle.com/c/quora-question-pairs/discussion/31019
https://github.com/qqgeogor/kaggle_quora_benchmark
"""

import logging

from os.path import join as join_path

from lib.project import project
from lib.dataset import load_train_df, load_test_df, Fields, FieldsTrain, FieldsTest
from lib.utils import makedirs

from sklearn.metrics import roc_auc_score

import distance


def jaccard(q1, q2):
    return distance.jaccard(q1.split(), q2.split())


def levenshtein1(q1, q2):
    return distance.nlevenshtein(q1, q2, method=1)


def levenshtein2(q1, q2):
    return distance.nlevenshtein(q1, q2, method=2)


def sorencen(q1, q2):
    return distance.sorensen(q1, q2)


def compute_quality(train_df, feature):
    corr = train_df[[FieldsTrain.is_duplicate, feature]].corr().values.tolist()
    auc = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)


def compute_features(train_df, test_df):

    train_df[Fields.jaccard] = train_df.apply(lambda r: jaccard(r[Fields.question1], r[Fields.question2]), axis=1)
    test_df[Fields.jaccard] = test_df.apply(lambda r: jaccard(r[Fields.question1], r[Fields.question2]), axis=1)
    quality_jaccard = compute_quality(train_df, Fields.jaccard)

    train_df[Fields.levenstein1] = train_df.apply(lambda r: levenshtein1(r[Fields.question1], r[Fields.question2]), axis=1)
    test_df[Fields.levenstein1] = test_df.apply(lambda r: levenshtein1(r[Fields.question1], r[Fields.question2]), axis=1)
    quality_levenstein1 = compute_quality(train_df, Fields.levenstein1)

    train_df[Fields.levenstein2] = train_df.apply(lambda r: levenshtein2(r[Fields.question1], r[Fields.question2]), axis=1)
    test_df[Fields.levenstein2] = test_df.apply(lambda r: levenshtein2(r[Fields.question1], r[Fields.question2]), axis=1)
    quality_levenstein2 = compute_quality(train_df, Fields.levenstein2)

    train_df[Fields.sorensen] = train_df.apply(lambda r: sorencen(r[Fields.question1], r[Fields.question2]), axis=1)
    test_df[Fields.sorensen] = test_df.apply(lambda r: sorencen(r[Fields.question1], r[Fields.question2]), axis=1)
    quality_sorensen = compute_quality(train_df, Fields.sorensen)

    quality = dict(
        jaccard=quality_jaccard,
        levenstein1=quality_levenstein1,
        levenstein2=quality_levenstein2,
        sorencen=quality_sorensen
    )

    return quality


def main(conf):
    dump_dir = conf['distances.dump.dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df(conf['distances.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['distances.dataset'])

    compute_features(train_df, test_df)

    logging.info('Writing train dataset to disk')
    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        Fields.jaccard,
        Fields.levenstein1,
        Fields.levenstein2,
        Fields.sorensen
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Writing test dataset to disk')
    test_df[[
        FieldsTest.test_id,
        Fields.jaccard,
        Fields.levenstein1,
        Fields.levenstein2,
        Fields.sorensen
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)
