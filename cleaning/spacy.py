'''
Note:

You can find .ipynb here: (which is more readable)
https://github.com/hubert0527/Kaggle-quora-question-pairs/blob/master/release_kernel.ipynb

'''

'''
▼ See the result at bottom directly if you're in a hurry ▼

(ctrl+F "final test")

'''

# ## Idea and motivation
#
# For cases like: 'Is Taiwan a good place to live ?'
# If we apply 'Taiwan' to word2vec directly, there are 3 cases might happen:
#
# 1. It works!
#
#         Taiwan is a very common word, so the word2vec method really captured its meaning in its vector space!
#
# 2. Somehow it works, but not very well.
#
#         Taiwan is not a very common word, so word2vec model cannot capture its meaning very well.
#
# 3. It fails.
#
#         Either the word is too rare or contains typo in it.
#
# And spaCy comes to me. It has an useful attribute "ent\_type\_" for each token, it tries to estimate the word's entity type in its pre-defined <a href="https://spacy.io/docs/api/annotation#named-entities">categories</a> based on the sentence's context.
#
# For example:
#
# ```
# testcase = 'Is Taiwan a good place to live ?'
# doc = nlp(testcase) # apply spacy on the sentense
# taiwan = doc[1] # word 'Taiwan' is at index 1
# print(taiwan.ent_type_)
#
# # yield "GPE" which means "Countries, cities, states."
# ```
#
# Then we can replace the original sentence.
#
# ```
# From: 'Is Taiwan a good place to live ?'
# To  : 'Is country a good place to live ?'
# ```
#
# The word "country"'s meaning is very likely captured better than the word 'Taiwan'. Although it losses some information, but the dataset becomes less noisy. It is kind of trade off, and I believe that adding a model trained on such dataset can increase ensembling result.
#
# ## Requirements
#
# 1. install spaCy and its corpus ( I use its en_core_web_md corpus )
#
# So, you can't run it directly here :(
#
#
# ## The attributes in spacy we'll use here
#
# 1. Token.lamma_
#
#         Get the token's base form
#         EX:
#             flies   => fly
#             am      => be
#             English => english
#
# 2. Token.ent_type_
#
#         Get the entity type of word
#         EX:
#             Taiwan    => GPE
#             English   => LANGUAGE
#             Hubert    => PERSON
#
# 3. Token.lower_
#
#         Get the lower case word
#         EX:
#             WHAT       => what
#             TensorFlow => tensorflow
#
# 4. Token.has_vector
#
#         See if the token can be identified by word2vec model used by spaCy (which is GloVe)
#         EX:
#             'we'              => True
#             'random_word_LOL' => False
#
# 5. Token.is_punct # not used (it can't identify '~' as punct ... )
#
#         If the token is punctuation
#         Ex:
#             '?'  => True
#             '我' => False
#
# ## Un-solved problems
#
# 1. We still have 3 special words can't have a meaningful vector.
#
#     SPECIAL_TOKENS = {
#         'quoted': 'quoted_item',            # words inside quotes or braces and can't be identified, ex: "What is 'blahblahblahblahblah' ?"
#         'non-ascii': 'non_ascii_word',      # as it means
#         'undefined': 'something'            # the word is either "too rare and spaCy can't label a ent_type_" or "typo"
#     }
#
#     Although each of them is very rare after the process, but we still need to deal with them ...
#
# 2. Typos are always our model's greatest enemy. Some other kernels has already mentioned about how to fix typos. If you can combine it with this kernel, the performance will probably increase.
#
#
# ## Some question might be asked
#
# 1. Umm.. So what is your score or improvement?
#
#         I'm still tuning my model. :(
#
# 2. In my experiance, the ent\_type\_ attribute result is noisy.
#
#         Yes, and that's why this kernel becomes so long.
#

# In[1]:

import spacy
import numpy as np
import pandas as pd

# In[2]:

nlp = spacy.load('en_core_web_md')

# In[3]:

print('Sanity test:')
doc = nlp(u'I am a good programmer!')
print([t for t in doc])

# In[ ]:




# ## Load dataset

# In[4]:

train_set = pd.read_csv('../quora-question-pairs/train.csv')

# In[5]:

qid_dict = {}

for i, series in train_set.iterrows():
    if series['qid1'] not in qid_dict:
        qid_dict[series['qid1']] = series['question1']
    if series['qid2'] not in qid_dict:
        qid_dict[series['qid2']] = series['question2']

# In[ ]:




# ## Define some special tokens that we'll use

# In[6]:

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

# In[ ]:




# ## Flags
#
# some hyper paramters can be tuned to generate different result
#
# ### [not implemented] is_substitute_proper_noun
#
# Now, for cases like : "is Taiwan a part of China?" => "is country a part of country"
#
# This simplifies question a lot, but loses a large portion of information.
#
# This is kind of "over cleaning" issue. May be normalize it using lemma is enough ?
#
# ### [not tested] is_remove_stopwords
#
# Removing stopwords seems can reduce some noise. But at the same time, we loses some informations.
#

# In[7]:

is_substitute_proper_noun = True  # a little complicated than it seems
is_remove_stopwords = False  # not yet tested

# In[8]:

if is_remove_stopwords:
    from nltk.corpus import stopwords

    stopwords = set(stopwords.words('english'))

# In[ ]:




# ## Clean up question text
#
# some part of the rules came from :
#
# https://www.kaggle.com/currie32/the-importance-of-cleaning-text

# In[9]:

import re


def clean_string(text):
    def pad_str(s):
        return ' ' + s + ' '

    # Empty question

    if type(text) != str or text == '':
        return ''

    # preventing first and last word being ignored by regex
    # and convert first word in question to lower case

    text = ' ' + text[0].lower() + text[1:] + ' '

    # replace all first char after either [.!?)"'] with lowercase
    # don't mind if we lowered a proper noun, it won't be a big problem

    def lower_first_char(pattern):
        matched_string = pattern.group(0)
        return matched_string[:-1] + matched_string[-1].lower()

    text = re.sub("(?<=[\.\?\)\!\'\"])[\s]*.", lower_first_char, text)

    # Replace weird chars in text

    text = re.sub("’", "'", text)  # special single quote
    text = re.sub("`", "'", text)  # special single quote
    text = re.sub("“", '"', text)  # special double quote
    text = re.sub("？", "?", text)
    text = re.sub("…", " ", text)
    text = re.sub("é", "e", text)

    # Clean shorthands

    text = re.sub("\'s", " ",
                  text)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"

    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

    # remove comma between numbers, i.e. 15,000 -> 15000

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    #     # all numbers should separate from words, this is too aggressive

    #     def pad_number(pattern):
    #         matched_string = pattern.group(0)
    #         return pad_str(matched_string)
    #     text = re.sub('[0-9]+', pad_number, text)

    # add padding to punctuations and special chars, we still need them later

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    def pad_pattern(pattern):
        matched_string = pattern.group(0)
        return pad_str(matched_string)

    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)

    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']),
                  text)  # replace non-ascii word with special word

    # indian dollar

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text

    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    # typos identified with my eyes

    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" undergraduation ", " undergraduate ", text)  # not typo, but GloVe can't find it
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)
    text = re.sub(r" begineer ", " beginner ", text)
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)
    text = re.sub(r" litrate ", " literate ", text)

    # for words like A-B-C-D or "A B C D",
    # if A,B,C,D individuaally has vector in glove:
    #     it can be treat as separate words
    # else:
    #     replace it as a special word, A_B_C_D is enough, we'll deal with that word later
    #
    # Testcase: 'a 3-year-old 4 -tier car'

    def dash_dealer(pattern):
        matched_string = pattern.group(0)
        splited = matched_string.split('-')
        splited = [sp.strip() for sp in splited if sp != ' ' and sp != '']
        joined = ' '.join(splited)
        parsed = nlp(joined)
        for token in parsed:
            # if one of the token is not common word, then join the word into one single word
            if not token.has_vector or token.text in SPECIAL_TOKENS.values():
                return '_'.join(splited)
        # if all tokens are common words, then split them
        return joined

    text = re.sub("[a-zA-Z0-9\-]*-[a-zA-Z0-9\-]*", dash_dealer, text)

    # try to see if sentence between quotes is meaningful
    # rule:
    #     if exist at least one word is "not number" and "length longer than 2" and "it can be identified by SpaCy":
    #         then consider the string is meaningful
    #     else:
    #         replace the string with a special word, i.e. quoted_item
    # Testcase:
    # i am a good (programmer)      -> i am a good programmer
    # i am a good (programmererer)  -> i am a good quoted_item
    # i am "i am a"                 -> i am quoted_item
    # i am "i am a programmer"      -> i am i am a programmer
    # i am "i am a programmererer"  -> i am quoted_item

    def quoted_string_parser(pattern):
        string = pattern.group(0)
        parsed = nlp(string[1:-1])
        is_meaningful = False
        for token in parsed:
            # if one of the token is meaningful, we'll consider the full string is meaningful
            if len(token.text) > 2 and not token.text.isdigit() and token.has_vector:
                is_meaningful = True
            elif token.text in SPECIAL_TOKENS.values():
                is_meaningful = True

        if is_meaningful:
            return string
        else:
            return pad_str(string[0]) + SPECIAL_TOKENS['quoted'] + pad_str(string[-1])

    text = re.sub('\".*\"', quoted_string_parser, text)
    text = re.sub("\'.*\'", quoted_string_parser, text)
    text = re.sub("\(.*\)", quoted_string_parser, text)
    text = re.sub("\[.*\]", quoted_string_parser, text)
    text = re.sub("\{.*\}", quoted_string_parser, text)
    text = re.sub("\<.*\>", quoted_string_parser, text)

    text = re.sub('[\(\)\[\]\{\}\<\>\'\"]', pad_pattern, text)

    # the single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(' s ', " ", text)

    # reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    text = text.strip()

    return text


# In[10]:

# %%time

for k in qid_dict:
    if k % 100000 == 0:
        print('Processed {} out of {}'.format(k, len(qid_dict)))
    qid_dict[k] = clean_string(qid_dict[k])

# In[ ]:




# ## process all questions in qid_dict using SpaCy

# In[11]:

# %%time

spacy_obj_dict = {}
total_len = len(qid_dict)
for i, k in enumerate(qid_dict):
    if i % 50000 == 0:
        print('Processed {} out of {}'.format(i, total_len))

    # some questions are null
    if type(qid_dict[k]) != str:
        qid_dict[k] = ''

    spacy_obj_dict[k] = nlp(qid_dict[k])

# In[ ]:




# In[ ]:




# In[ ]:




# ## Replace proper nouns in sentence to related types
#
# i.e.
#
#     'Google'  -> 'organization'
#     'Spanish' -> 'language'
#

# In[12]:

# These are types defined by spaCy, replace proper nouns with a related common word
# Since PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL are sharing very close meaning and is frequently being mis-classified,
#     so I decided to replace all of them with "number"

ENTITY_ENUM = {
    '': '',
    'PERSON': 'person',
    'NORP': 'nationality',
    'FAC': 'facility',
    'ORG': 'organization',
    'GPE': 'country',
    'LOC': 'location',
    'PRODUCT': 'product',
    'EVENT': 'event',
    'WORK_OF_ART': 'artwork',
    'LANGUAGE': 'language',
    'DATE': 'date',
    'TIME': 'time',
    #     'PERCENT': 'percent',
    #     'MONEY': 'money',
    #     'QUANTITY': 'quantity',
    #     'ORDINAL': 'ordinal',
    #     'CARDINAL': 'cardinal',
    'PERCENT': 'number',
    'MONEY': 'number',
    'QUANTITY': 'number',
    'ORDINAL': 'number',
    'CARDINAL': 'number',
    'LAW': 'law'
}

NUMERIC_TYPES = set([
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'QUANTITY',
    'ORDINAL',
    'CARDINAL',
])

# In[ ]:




# ## Isn't this vey easy ? Why you write these dirty codes?
#
# here's a fast example:

# In[13]:

doc1 = nlp('Is Hubert a good programmer?')
doc2 = nlp('Hubert is eating bugs!')

print(doc1[1].text, '->', doc1[1].ent_type_)  # can't recognize entity type
print(doc2[0].text, '->', doc2[0].ent_type_)  # can recognize entity type is a person

# ### What !?
#
# Yep, spaCy is powerful, but it doesn't always work. Its entity type prediction is noisy. So, here comes my solution:
#
# Let's go through the whole dataset first. Then we'll come back and tell you what this proper noun is in most cases.
#
# ### Benefit
#
# 1. Very stable
# 2. The result is good for those proper nouns which appear many times in the training set
#
# ### Disadvantages
#
# 1. The word has very different meaning in different cases might suck. EX:
#
#         "Can't you google that ? " => ['can', 'not', 'you', 'organization', 'that']
#         "I want to work in Google" => ['i', 'want', 'to', 'work', 'in', 'organization']
#
# 2. If a word A is a proper noun, but spaCy can't recognize its entity type in most cases. We have no clue to identify if the word is a very common word or a very rare word.
#
#         As a solution, I also recorded the second highest voted entity type.
#         If a word is abolutely a proper noun, we can choose to use its second highest entity type.
#
#
#
#

# In[ ]:




# ## Go through all questions and records entity type of all words

# In[14]:

# if the second type count is extremely low, consider it as noise and the word has no second type
threshold_of_second_type = 3

# In[15]:

# %%time

vote_dict = {}
word_ent_type_dict = {}
word_ent_type_second_dict = {}

# construct vote dictionary

for qid in spacy_obj_dict:

    if qid % 100000 == 0:
        print('Processing {} / {}'.format(qid, len(spacy_obj_dict)))

    for token in spacy_obj_dict[qid]:

        if token.lower_ not in vote_dict:
            vote_dict[token.lower_] = {}

        if token.ent_type_ not in vote_dict[token.lower_]:
            vote_dict[token.lower_][token.ent_type_] = 0

        # if the token has_vector is True, maybe we shouldn't record its

        vote_dict[token.lower_][token.ent_type_] += 1  # TODO: not sure if storing in lowercase form is safe ?

# vote for what should the type be

for key in vote_dict:

    # non-type has lower priority
    if '' in vote_dict[key]:
        vote_dict[key][''] = vote_dict[key][''] - 0.1

    ents = list(vote_dict[key].keys())
    bi_list = [
        ents,
        [vote_dict[key][ent] for ent in ents]
    ]

    # if several ent_type_ have same count, just let it go, making them share same ent_type_ is enough
    # TODO: if have time, can design a better metric to deal with second graded type
    sorted_idx = np.argsort(bi_list[1])

    if sorted_idx.shape[0] > 1:
        best_idx = sorted_idx[-1]
        second_idx = sorted_idx[-2]
        word_ent_type_dict[key] = bi_list[0][best_idx]
        if bi_list[1][second_idx] > threshold_of_second_type:
            word_ent_type_second_dict[key] = bi_list[0][second_idx]
    else:
        best_idx = sorted_idx[-1]
        word_ent_type_dict[key] = bi_list[0][best_idx]


# In[16]:

# APIs to access entity type

def token_type_lookup(token, report_detail=False):
    if type(token) == str:
        token = nlp(token)[0]

    key = token.lower_

    try:
        if report_detail:
            print(ENTITY_ENUM[word_ent_type_dict[key]], ' <= ',
                  {ENTITY_ENUM[ent_t]: vote_dict[key][ent_t] for ent_t in vote_dict[key]})

        return word_ent_type_dict[key]

    except KeyError:
        return ''


def is_token_has_second_type(token):
    if type(token) == str:
        token = nlp(token)[0]

    key = token.lower_

    try:
        return key in word_ent_type_second_dict
    except KeyError:
        return False


def token_second_type_lookup(token, report_detail=False):
    if type(token) == str:
        token = nlp(token)[0]

    key = token.lower_

    try:
        if report_detail:
            print(ENTITY_ENUM[word_ent_type_second_dict[key]], ' <= ',
                  {ENTITY_ENUM[ent_t]: vote_dict[key][ent_t] for ent_t in vote_dict[key]})

        return word_ent_type_second_dict[key]
    except KeyError:
        return ''


# ### quick tests

# In[17]:

testcases = [
    'ms',
    'taiwanese',
    'google',
    'tensorflow',
    'mac'
]

for testcase in testcases:
    print(testcase)
    token_type_lookup(testcase, report_detail=True)
    print()

# In[ ]:




# ## Start to clean up questions with spaCy

# In[18]:

exception_list = set(['need'])  # spaCy identifies need's lamma as 'ne', which is not we want
numeric_types = set(['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])

'''
Pseudo code:

for each token:

    if token is in exception_list:
        use its original form (i.e. "need" can be parsed as "ne" ... we absolutely will use "need")

    elif token is in SPECIAL_TOKENS:
        if token's entity type can be identified:
            use the ent_type_ detected
        else:
            use the special token directly

    elif token is a special character or not cleaned ' ' or 's':
        skip it

    # cuz we don't want a single proper noun being expended into several words
    # ex:
    # 'meet you on 2017-02-31'
    # (O) => meet you on date
    # (X) => meet you on date date date
    elif ( the current ent_type_ and previous ent_type_ are both numeric types in NUMERIC_TYPES ) or ( the current ent_type_ == previous ent_type_ ):
        skip it

    elif ( token is labeled with an ent_type_ in word_ent_type_dict ):
        replace the token with ent_type_ specified in word_ent_type_dict

    elif ( the transform only changes uppercases to lowercases ) and ( the changes happened not only on the first char ):
        then it is a proper noun (in most cases) , we'll try to pick the largest ent_type_ in word_ent_type_dict which is not ''

    # i.e. the token's ent_type_ cannot be found, is either a very common word or a very rare word
    else:
        if token's lemma == '-PRON-':
            it's a weird case, just use its original form
        elif token's lemma has_vector is true: # i.e. can be identified by GloVe
            use the lemma form
        elif the token has second ent_type_ usable:
            use the second ent_type_
        elif the token itself has_vector is true:
            use the token's raw text
        # have no idea what happenned, most cases are "very rare proper noun" or "type"
        else:
            use "something as a replacement"

'''


def process_question_with_spacy(spacy_obj, debug=False, show_fail=False, idx=None):
    def not_alpha_or_digit(token):
        ch = token.text[0]
        return not (ch.isalpha() or ch.isdigit())

    result_word_list = []
    res = ''

    # for continuous entity type string, we need only single term.
    # EX: "2017-01-01"
    # => "time time time" (X)
    # => "time" (O)
    previous_ent_type = None

    is_a_word_parsed_fail = False
    fail_words = []

    for token in spacy_obj:

        global_ent_type = token_type_lookup(token)

        # problematic token, use its base form
        if token.text in exception_list:

            previous_ent_type = None
            result_word_list.append(token.text)

        # special kind of tokens
        elif token.text in SPECIAL_TOKENS.values():

            # we have no choice but use the entity type detected by spaCy directly, but it still might fail
            if token.ent_type_ != '':
                previous_ent_type = token.ent_type_
                result_word_list.append(token.ent_type_)

            else:
                previous_ent_type = None
                result_word_list.append(token.text)


        # skip none words tokens
        elif not_alpha_or_digit(token) or token.text == ' ' or token.text == 's':
            previous_ent_type = None
            if debug: print(token.text, ' : remove punc or special chars')


        # if the "remove stop word" flag is set to True
        elif is_remove_stopwords and token.lemma_ in is_remove_stopwords:
            previous_ent_type = None
            if debug: print(token.text, ' : remove stop word')


        # contiguous same type, skip it
        elif global_ent_type == previous_ent_type or token.ent_type_ == previous_ent_type:
            if debug: print('contiguous same type')
        elif global_ent_type in NUMERIC_TYPES and previous_ent_type in NUMERIC_TYPES:
            if debug: print('contiguous numeric')
        elif token.ent_type_ in NUMERIC_TYPES and previous_ent_type in NUMERIC_TYPES:
            if debug: print('contiguous numeric')


        # number without an ent_type_
        elif token.text.isdigit():

            if debug: print(token.text, 'force to be number')

            if previous_ent_type in NUMERIC_TYPES:
                pass
            else:
                previous_ent_type = 'CARDINAL'  # any number type would be okay
                result_word_list.append('number')


        # replace proper nouns into name entities.
        # EX:
        # Original : Taiwan is next to China
        # Result   : country is next to country
        elif global_ent_type != '':

            result_word_list.append(ENTITY_ENUM[global_ent_type])
            previous_ent_type = global_ent_type
            if debug: print(token.text, ' : sub ent_type:', ENTITY_ENUM[global_ent_type])


        # Identify if a word is proper noun or not, if it is a proper noun, we'll try to use second highest rated ent_type_
        #
        # A proper noun has following special patterns:
        #     1. its lemma_ (base form) returned by spaCy is just its lowercase form
        #     2. if one of its character except the first character is uppercase, it is a propernoun (in most cases)
        # except the special cases like "I LOVE YOU", we cal say that if (1.) and (2.), then the token is proper noun
        #
        # for cases like "Tensorflow", we have no good rule to identify it is a proper noun or not ... let's just move on
        elif token.lower_ == token.lemma_ and token.text[1:] != token.lemma_[1:] and is_token_has_second_type(token):
            second_type = token_second_type_lookup(token)
            result_word_list.append(ENTITY_ENUM[second_type])
            if debug: print(token.text, ' : use second ent_type:', ENTITY_ENUM[second_type])
            previous_ent_type = second_type


        # words arrive here are either "extremely common" or "extremely rare and has no method to deal with"
        else:
            # A weird behavior of SpaCy, it substitutes [I, my, they] into '-PRON-', which mean pronoun (代名詞)
            # More detail in : https://github.com/explosion/spaCy/issues/962
            if token.lemma_ == '-PRON-':
                result_word_list.append(token.lower_)
                res = token.lower
                previous_ent_type = None

            # the lemma can be identified by GloVe
            elif nlp(token.lemma_)[0].has_vector:
                result_word_list.append(token.lemma_)
                res = token.lemma_
                previous_ent_type = None

            # the lemma cannot be identified, very probably a proper noun
            elif is_token_has_second_type(token):
                second_type = token_second_type_lookup(token)
                result_word_list.append(ENTITY_ENUM[second_type])
                res = ENTITY_ENUM[second_type]
                previous_ent_type = second_type
                if debug: print(token.text, ' : use second ent_type in else :', ENTITY_ENUM[second_type])

            # the lemma is not in glove and Spacy can't identify if it is a proper noun, last try,
            #      if the word itself can be identified by GloVe or not
            elif nlp(token.lower_)[0].has_vector:
                result_word_list.append(token.lower_)
                res = token.lower_
                previous_ent_type = None
                if debug: print(token.text, ' : the token itself can be identified :', token.lower_)
            elif token.has_vector:
                result_word_list.append(token.text)
                res = token.text
                previous_ent_type = None
                if debug: print(token.text, ' : the token itself can be identified :', token.text)

            # Damn, I have totally no idea what's going on
            # You got to deal with it by yourself
            # In my case, I use fasttext to deal with it
            else:
                is_a_word_parsed_fail = True
                fail_words.append(token.text)
                previous_ent_type = None

                #  Question:
                #  can we replace all this kind of word into "something" ?
                result_word_list.append(SPECIAL_TOKENS['undefined'])
                if debug: print(token.text, ' : can\'t identify, replace with "something"')

    if show_fail and is_a_word_parsed_fail:
        if idx != None:
            print('At qid=', idx)
        print('Fail words: ', fail_words)
        print('Before:', spacy_obj.text)
        print('After: ', ' '.join(result_word_list))
        print('====================================================================')

    return np.array(result_word_list)


# ### show part of the result
#
# you can see most of them are typo and rare words

# In[19]:

for k in range(1, 300):
    process_question_with_spacy(spacy_obj_dict[k], show_fail=True, idx=k)

# In[20]:

# %%time
qid_spacy_cleaned_question_dict = {}
qid_spacy_cleaned_word_lists_dict = {}

for i, k in enumerate(spacy_obj_dict):

    if i % 50000 == 0:
        print('Processed {} out of {}'.format(i, total_len))

    word_list = process_question_with_spacy(spacy_obj_dict[k])
    qid_spacy_cleaned_word_lists_dict[k] = word_list
    qid_spacy_cleaned_question_dict[k] = ' '.join(word_list)


# In[ ]:




# ### create a method to process any input string

# In[21]:

# This is the summary of this kernel, so simple LOL
def process_new_string(s):
    s = clean_string(s)
    s = nlp(s)
    s = process_question_with_spacy(s)
    return s


# In[ ]:




# ## Custom testcases, seems works not bad but also not  well LOL
# final test

# In[24]:

testcases = [

    # testcases writen by me or randomly copy-paste from websites

    'Hi! I\'m Hubert, I am new to Kaggle and this is my first Kernel ! Hope you like It ~ ',
    "spaCy excels at large-scale information extraction tasks. It's written from the ground up in carefully memory-managed Cython. Independent research has confirmed that spaCy is the fastest in the world. If your application needs to process entire web dumps, spaCy is the library you want to be using.",
    'Google began in January 1996 as a research project by Larry Page and Sergey Brin when they were both PhD students at Stanford University in Stanford, California.[6]',

    # some weird testcases

    '3-year-old child can also be a good-programmer',
    'what "Пока" means ?',
    'what Пока means ?',
    'IS Taiwan a Good place to Live ? ',
    'How to being hired by google?',
    'I ran 3mm\/sec',

    # some testcases random pick from training set

    # Mark is detected as not a proper noun ... Poor Mark
    'Is trump a billionaire? How do we know for sure? Why do most people (Mark Cuban excepted) just take him at his word on this?',
    'Is it better to have a career in C, C++ or in Node.js?',
    'Which is better intel i5 (6th gen) or i7 (5th gen)?',
    "Which is correct: '2 dozen of eggs cost 30 rupees' or '2 dozen of eggs costs 30 rupees'?",
]

for testcase in testcases:
    print(testcase)
    print('=> ' + ' '.join(process_new_string(testcase)))
    print(
    '\n========================================================================================================\n')

'''
Outputs:

Hi! I'm Hubert, I am new to Kaggle and this is my first Kernel ! Hope you like It ~
=> hi i be hubert i be new to something and this be my number kernel hope you like it

========================================================================================================

spaCy excels at large-scale information extraction tasks. It's written from the ground up in carefully memory-managed Cython. Independent research has confirmed that spaCy is the fastest in the world. If your application needs to process entire web dumps, spaCy is the library you want to be using.
=> spacy excel at large scale information extraction task it write from the ground up in carefully memory manage cython independent research have confirm that spacy be the fast in the world if your application need to process entire web dump spacy be the library you want to be use

========================================================================================================

Google began in January 1996 as a research project by Larry Page and Sergey Brin when they were both PhD students at Stanford University in Stanford, California.[6]
=> organization begin in date as a research project by person and person when they be both artwork student at organization in organization country quoted_item

========================================================================================================

3-year-old child can also be a good-programmer
=> number child can also be a good programmer

========================================================================================================

what "Пока" means ?
=> what non_ascii_word mean

========================================================================================================

what Пока means ?
=> what non_ascii_word mean

========================================================================================================

IS Taiwan a Good place to Live ?
=> be country a good place to live

========================================================================================================

How to being hired by google?
=> how to be hire by organization

========================================================================================================

I ran 3mm\/sec
=> i run number sec

========================================================================================================

Is trump a billionaire? How do we know for sure? Why do most people (Mark Cuban excepted) just take him at his word on this?
=> be person a billionaire how do we know for sure why do most people mark person except just take him at his word on this

========================================================================================================

Is it better to have a career in C, C++ or in Node.js?
=> be it good to have a career in c c or in person js

========================================================================================================

Which is better intel i5 (6th gen) or i7 (5th gen)?
=> which be good organization i5 number gen or i7 number gen

========================================================================================================

Which is correct: '2 dozen of eggs cost 30 rupees' or '2 dozen of eggs costs 30 rupees'?
=> which be correct number of egg cost number or number of egg cost number

========================================================================================================

'''




# ### Hope you like my approach, if so, please give me an upvote :)
#
# My approach consists of some heuristic and may have risk on 'over cleaning'.
#
# If you have any great idea to improve it or find some of my moves is problematic. Please leave a comment, I appreciate that !

# In[ ]:



