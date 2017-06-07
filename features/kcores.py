# -*- coding: utf-8 -*-

import logging

from os.path import join as join_path

import pandas as pd

from lib.project import project
from lib.dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest
from lib.utils import makedirs


def load_kcores(filename):
    df = pd.read_csv(filename)
    data = {}

    def update(r):
        data[r['question']] = r['kcores']

    df.apply(update, axis=1)
    return data


def main(conf):
    logging.info('Loading training dataset')
    train_df = load_train_df(conf['kcores.dataset'])

    logging.info('Loading test dataset')
    test_df = load_test_df(conf['kcores.dataset'])

    logging.info('Loading kcores dump')
    kcores = load_kcores(conf['kcores.source'])

    def substitute_kcores(q):
        return kcores.get(q.lower(), 0)

    train_df['q1_kcores'] = train_df.apply(lambda r: substitute_kcores(r['question1']), axis=1)
    train_df['q2_kcores'] = train_df.apply(lambda r: substitute_kcores(r['question2']), axis=1)

    test_df['q1_kcores'] = test_df.apply(lambda r: substitute_kcores(r['question1']), axis=1)
    test_df['q2_kcores'] = test_df.apply(lambda r: substitute_kcores(r['question2']), axis=1)

    logging.info('Writing dump')
    dump_dir = conf['kcores.dump.dir']
    makedirs(dump_dir)

    train_df[[
        FieldsTrain.id,
        FieldsTrain.is_duplicate,
        FieldsTrain.q1_kcores,
        FieldsTrain.q2_kcores
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    test_df[[
        FieldsTest.test_id,
        FieldsTest.q1_kcores,
        FieldsTest.q2_kcores
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    main(project().conf)
