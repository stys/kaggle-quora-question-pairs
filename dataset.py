import codecs
from os.path import join

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from project import conf


def _load_df(filename):
    with codecs.open(filename) as f:
        return pd.read_csv(f).fillna(conf['dataset.fillna'])


def load_train_df():
    return _load_df(join(conf['dataset.dir'], conf['dataset.train']))


def load_test_df():
    return _load_df(join(conf['dataset.dir'], conf['dataset.test']))


def skfold():
    return StratifiedKFold(conf['cv.nfolds'], shuffle=False, random_state=conf['cv.seed'])


def submission(filename, test_ids, predictions):
    df = pd.DataFrame()
    df['test_id'] = test_ids
    df['is_duplicate'] = predictions
    df.to_csv(filename, index=False)


class Fields(object):
    question1 = 'question1'
    question2 = 'question2'

    # fs-1
    len_q1 = 'len_q1'
    len_q2 = 'len_q2'
    diff_len = 'diff_len'
    len_char_q1 = 'len_char_q1'
    len_char_q2 = 'len_char_q2'
    diff_len_char = 'diff_len_char'
    len_word_q1 = 'len_word_q1'
    len_word_q2 = 'len_word_q2'
    diff_len_word = 'diff_len_word'


class FieldsTrain(Fields):
    id = 'id',
    qid1 = 'qid1',
    qid2 = 'qid2',
    is_duplicate = 'is_duplicate'
    cv_fold = 'cv_fold'
    linear_word = 'linear_words'
    linear_char = 'linear_char'


class FieldsTest(Fields):
    test_id = 'test_id'
    linear_word_cv = 'linear_word_cv'
    linear_word_full = 'linear_word_full'
    linear_word_full_weighted = 'linear_word_full_weighted'
    linear_char_cv = 'linear_char_cv'
    linear_char_full = 'linear_char_full'
    linear_char_full_weighted = 'linear_char_full_weighted'
