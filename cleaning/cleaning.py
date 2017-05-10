# -*- coding: utf-8 -*-


import logging
import re

from os.path import join as join_path
from collections import namedtuple

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

from lib.dataset import load_train_df, load_test_df, Fields
from lib.utils import makedirs

stopwords = set(nltk_stopwords.words("english"))


class Substitution(namedtuple('Substitution', ['pattern', 'repl'])):
    pass

substitutions = [
    Substitution(re.compile("[^A-Za-z0-9^,!./'+-=]"), " "),
    Substitution(re.compile("what's"), "what is"),
    Substitution(re.compile("\'s"), " "),
    Substitution(re.compile("\'ve"), " "),
    Substitution(re.compile("can't"), "cannot "),
    Substitution(re.compile("n't"), " not "),
    Substitution(re.compile("i'm"), "i am "),
    Substitution(re.compile("\'re"), " are "),
    Substitution(re.compile("\'d"), " would "),
    Substitution(re.compile("\'ll"), " will "),
    Substitution(re.compile(","), " "),
    Substitution(re.compile("\."), " "),
    Substitution(re.compile("!"), " ! "),
    Substitution(re.compile("/"), " "),
    Substitution(re.compile("\^"), " ^ "),
    Substitution(re.compile("\+"), " + "),
    Substitution(re.compile("\-"), " - "),
    Substitution(re.compile("="), " = "),
    Substitution(re.compile("'"), " "),
    Substitution(re.compile("(\d+)(k)"), "\g<1>000"),
    Substitution(re.compile(":"), " : "),
    Substitution(re.compile(" e g "), " eg "),
    Substitution(re.compile(" b g "), " bg "),
    Substitution(re.compile(" u s "), " american "),
    Substitution(re.compile("\0s"), "0"),
    Substitution(re.compile(" 9 11 "), "911"),
    Substitution(re.compile("e - mail"), "email"),
    Substitution(re.compile("e-mail"), "email"),
    Substitution(re.compile("j k"), "jk"),
    Substitution(re.compile("\s{2,}"), " ")
]

stemmer = SnowballStemmer('english')


def clean(text, remove_stopwords=False, stem_words=False, **other):
    text = text.lower().split()

    if remove_stopwords:
        text = [w for w in text if w not in stopwords]

    text = ' '.join(text)

    for subst in substitutions:
        text = subst.pattern.sub(subst.repl, text)

    if stem_words:
        text = text.split()
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


def main(conf):
    dump_dir = conf['cleaning']['dump']['dir']
    makedirs(dump_dir)

    logging.info('Loading train dataset')
    train_df = load_train_df()

    logging.info('Cleaning train dataset')
    train_df[Fields.question1] = train_df[Fields.question1].apply(lambda q: clean(q, **conf['cleaning']))
    train_df[Fields.question2] = train_df[Fields.question2].apply(lambda q: clean(q, **conf['cleaning']))

    logging.info('Writing train dataset')
    train_df.to_csv(join_path(dump_dir, 'train.csv'), index=False)

    logging.info('Loading test dataset')
    test_df = load_test_df()

    logging.info('Cleaning test dataset')
    test_df[Fields.question1] = test_df[Fields.question1].apply(lambda q: clean(q, **conf['cleaning']))
    test_df[Fields.question2] = test_df[Fields.question2].apply(lambda q: clean(q, **conf['cleaning']))

    logging.info('Writing test dataset')
    test_df.to_csv(join_path(dump_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    import project
    main(conf=project.conf)
