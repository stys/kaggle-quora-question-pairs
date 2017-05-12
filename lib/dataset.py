import codecs
from os.path import join

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lib.project import project
conf = project().conf


def _load_df(filename, fillna):
    with codecs.open(filename) as f:
        return pd.read_csv(f).fillna(fillna)


def load_train_df(conf):
    return _load_df(join(conf['dir'], conf['train']), conf['fillna'])


def load_test_df(conf):
    return _load_df(join(conf['dir'], conf['test']), conf['fillna'])


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

    # word2vec
    w2v_wmd = 'w2v_wmd'
    w2v_wmd_norm = 'w2v_wmd_norm'
    w2v_cos = 'w2v_cos'
    w2v_city = 'w2v_city'
    w2v_jacc = 'w2v_jacc'
    w2v_canb = 'w2v_canb'
    w2v_eucl = 'w2v_eucl'
    w2v_mink = 'w2v_mink'
    w2v_bray = 'w2v_bray'
    w2v_skew_q1 = 'w2v_skew_q1'
    w2v_skew_q2 = 'w2v_skew_q2'
    w2v_kurt_q1 = 'w2v_kurt_q1'
    w2v_kurt_q2 = 'w2v_kurt_q2'


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
