import errno
import logging
from os import makedirs
from os.path import join as join_path

from lib.project import project

from lib.dataset import load_train_df, load_test_df
from lib.dataset import Fields, FieldsTrain, FieldsTest
from fuzzywuzzy import fuzz

from sklearn.metrics import roc_auc_score


def compute_quality(train_df, feature):
    corr = train_df[[FieldsTrain.is_duplicate, feature]].corr().values.tolist()
    auc = roc_auc_score(train_df[FieldsTrain.is_duplicate], train_df[feature])
    logging.info('Feature %s: CORR=%s, AUC=%s', feature, corr, auc)
    return dict(corr=corr, auc=auc)


def compute_features(train_df, test_df):

    train_df[Fields.qratio] = train_df.apply(
        lambda row: fuzz.QRatio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.qratio] = test_df.apply(
        lambda row: fuzz.QRatio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_qratio = compute_quality(train_df, Fields.qratio)

    train_df[Fields.wratio] = train_df.apply(
        lambda row: fuzz.WRatio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.wratio] = test_df.apply(
        lambda row: fuzz.WRatio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_wratio = compute_quality(train_df, Fields.wratio)

    train_df[Fields.partial_ratio] = train_df.apply(
        lambda row: fuzz.partial_ratio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.partial_ratio] = test_df.apply(
        lambda row: fuzz.partial_ratio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_partial_ratio = compute_quality(train_df, Fields.partial_ratio)

    train_df[Fields.partial_token_set_ratio] = train_df.apply(
        lambda row: fuzz.partial_token_set_ratio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.partial_token_set_ratio] = test_df.apply(
        lambda row: fuzz.partial_token_set_ratio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_partial_token_set_ratio = compute_quality(train_df, Fields.partial_token_set_ratio)

    train_df[Fields.partial_token_sort_ratio] = train_df.apply(
        lambda row: fuzz.partial_token_sort_ratio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.partial_token_sort_ratio] = test_df.apply(
        lambda row: fuzz.partial_token_sort_ratio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_partial_token_sort_ratio = compute_quality(train_df, Fields.partial_token_sort_ratio)

    train_df[Fields.token_set_ratio] = train_df.apply(
        lambda row: fuzz.token_set_ratio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.token_set_ratio] = test_df.apply(
        lambda row: fuzz.token_set_ratio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_token_set_ratio = compute_quality(train_df, Fields.token_set_ratio)

    train_df[Fields.token_sort_ratio] = train_df.apply(
        lambda row: fuzz.token_sort_ratio(str(row[FieldsTrain.question1]), str(row[FieldsTrain.question2])), axis=1)
    test_df[Fields.token_sort_ratio] = test_df.apply(
        lambda row: fuzz.token_sort_ratio(str(row[FieldsTest.question1]), str(row[FieldsTest.question2])), axis=1)
    quality_token_sort_ratio = compute_quality(train_df, Fields.token_sort_ratio)

    quality = dict(
        quality_qratio=quality_qratio,
        quality_wratio=quality_wratio,
        quality_partial_ratio=quality_partial_ratio,
        quality_partial_token_set_ratio=quality_partial_token_set_ratio,
        quality_partial_token_sort_ratio=quality_partial_token_sort_ratio,
        quality_token_set_ratio=quality_token_set_ratio,
        quality_token_sort_ratio=quality_token_sort_ratio
    )

    return quality


def main(conf):
    dump_dir = conf['fuzzy.dump.dir']

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
        Fields.qratio,
        Fields.wratio,
        Fields.partial_ratio,
        Fields.partial_token_set_ratio,
        Fields.partial_token_sort_ratio,
        Fields.token_set_ratio,
        Fields.token_sort_ratio
    ]].to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Writing test dataset to disk')
    test_df[[
        FieldsTest.test_id,
        Fields.qratio,
        Fields.wratio,
        Fields.partial_ratio,
        Fields.partial_token_set_ratio,
        Fields.partial_token_sort_ratio,
        Fields.token_set_ratio,
        Fields.token_sort_ratio
    ]].to_csv(join_path(dump_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main(project().conf)