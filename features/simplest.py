# -*- coding: utf-8 -*-

"""
Simplest features from Abhishek's fs-1
https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
"""

from dataset import Fields


def features(train_df, test_df):
    train_df[Fields.len_q1] = train_df[Fields.question1].map(lambda q: len(q))
    test_df[Fields.len_q1] = test_df[Fields.question1].map(lambda q: len(q))

    train_df[Fields.len_q2] = train_df[Fields.question2].map(lambda q: len(q))
    test_df[Fields.len_q2] = test_df[Fields.question2].map(lambda q: len(q))

    train_df[Fields.diff_len] = abs(train_df[Fields.len_q1] - train_df[Fields.len_q2])
    test_df[Fields.diff_len] = abs(test_df[Fields.len_q1] - test_df[Fields.len_q2])

    # TODO


def main(conf):
    pass


if __name__ == '__main__':
    import project
    main(project.conf)
