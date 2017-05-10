import codecs
from os.path import join

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lib.project import project
conf = project().conf


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

    # counters
    freq_q1 = 'freq_q1'
    freq_q2 = 'freq_q2'

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
    id = 'id'
    qid1 = 'qid1'
    qid2 = 'qid2'
    is_duplicate = 'is_duplicate'
    linear = 'linear'


class FieldsTest(Fields):
    test_id = 'test_id'
    linear_cv = 'linear_cv'
    linear_full = 'linear_full'
    linear_full_weighted = 'linear_full_weighted'
