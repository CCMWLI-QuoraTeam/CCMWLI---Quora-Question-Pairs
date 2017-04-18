
import spacy
from string import punctuation
import pickle
import numpy as np
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook
import os
from spacy.en import English

nlp = spacy.load('en')

def basic_preproc(questions1, questions2, filepath, use_cached=False):

    if use_cached:
        with open(filepath, 'rb') as data_file:
            data = pickle.load(data_file)
        return data[:,1], data[:,2]

    n_questions = len(questions1)
    pre_q1 = []
    pre_q2 = []

    for q1,q2 in zip(questions1,questions2):

        if len(pre_q1) % 10000 == 0:
            progress = len(pre_q1) / n_questions * 100
            print("Basic preprocessing is {}% complete.\r".format(progress)),

        # All to lowercase
        q1 = q1.lower()
        q2 = q2.lower()
        # Remove punctuation
        q1 = ''.join([c for c in q1 if c not in punctuation])
        q2 = ''.join([c for c in q2 if c not in punctuation])
        # TODO Proposal: Fix spelling mistakes. Usefull link: https://www.quora.com/Natural-Language-Processing-APIs-for-common-mispellings
        # Remove non-english words
        q1 = rm_stop_words(q1)
        q2 = rm_stop_words(q2)
        # Steaming
        q1 = lemmatize(q1)
        q2 = lemmatize(q2)
        # Save
        pre_q1.append(q1)
        pre_q2.append(q2)

    # Save data
    final_data = (pre_q1, pre_q2)
    with open(filepath, 'wb') as outfile:
        pickle.dump(final_data, outfile)

    return pre_q1, pre_q2

def rm_stop_words(text):
    text = text.split()
    text = [w for w in text if not nlp.vocab[w].is_stop]
    text = " ".join(text)
    return text

def lemmatize(text):
    text = nlp(text)
    stemmed_words = [word.lemma_ for word in text]
    text = " ".join(stemmed_words)
    return text