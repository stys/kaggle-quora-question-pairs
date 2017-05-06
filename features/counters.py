# -*- coding: utf-8 -*-

"""
Question frequencies are correlated with duplicate probability
"""

import logging

import pandas as pd
from dataset import load_train_df, load_test_df, FieldsTrain, FieldsTest
from sklearn.metrics import roc_auc_score


def counters(train_df, test_df):
    # build collection of all unique questions
    questions = pd.concat([
            train_df[[FieldsTrain.question1]].rename(columns={'question1': 'q'}),
            train_df[[FieldsTrain.question2]].rename(columns={'question2': 'q'}),
            test_df[[FieldsTest.question1]].rename(columns={'question1': 'q'}),
            test_df[[FieldsTest.question2]].rename(columns={'question2': 'q'})
        ],
        ignore_index=True,
    )

    qdict = questions['q'].value_counts().to_dict()

    train_df['q1_freq'] = train_df[FieldsTrain.question1].map(lambda q: qdict.get(q, 0))
    train_df['q2_freq'] = train_df[FieldsTrain.question2].map(lambda q: qdict.get(q, 0))

    print train_df[['q1_freq', 'q2_freq', 'is_duplicate']].corr()

    logging.info("Frequency of question1 AUC=%s", roc_auc_score(train_df['is_duplicate'], train_df['q1_freq']))
    logging.info("Frequency of question2 AUC=%s", roc_auc_score(train_df['is_duplicate'], train_df['q2_freq']))

    # Compute CTR's by question



def main(conf):
    train_df = load_train_df()
    test_df = load_test_df()

    counters(train_df, test_df)


if __name__ == '__main__':
    import project
    main(project.conf)
