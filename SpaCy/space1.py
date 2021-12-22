# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:32:13 2021
playing around with spacy library
@author: starg
"""

import spacy
import re
import pandas as pd
import string
import textacy
from whatlies.language import SpacyLanguage
from spacy.language import Language
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#applying this function to a document will tokenize the text, apply a POS tag to each token,
#calculate syntactic dependencies, scan for named entities e.g. Netflix and then classify text
nlp = spacy.load("en_core_web_md")
lang = SpacyLanguage("en_core_web_md")
#nlp.Defaults.stop_words.remove('no')
extra_stops = ['netflix','customer','service','account','movies','like','no']
nlp.Defaults.stop_words.update(extra_stops)




# create textacy model using large spaCy model
en = textacy.load_spacy_lang("en_core_web_md")
df = pd.read_csv(r'../Selenium/reviewdata.csv',header=[0])
stopList = nlp.Defaults.stop_words
df.columns=['DATE','REVIEW','RATING']
df.dropna(subset =["DATE","REVIEW"],inplace=True)
df = df.sort_index()

texts = {
    "text": df.REVIEW,
    "date": df.DATE,
    "rating": df.RATING
}
corpus = textacy.Corpus(lang=en, data=texts)
# check corpus stats
corpus.n_docs, corpus.n_sents, corpus.n_tokens
def clean_text_round1(text):
    #lowercase
    text = text.lower()
    #replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    #remove instances of punctuation
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #remove numbers and words attached to numbers
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

data_clean = pd.DataFrame(df.REVIEW.apply(round1))

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\r', '', text)
    text = re.sub(' nt ','',text)
    text = re.sub(' ve ','',text)
    return text

round2 = lambda x: clean_text_round2(x)

data_clean = pd.DataFrame(data_clean.REVIEW.apply(round2))


docs = [nlp(i) for i in data_clean['REVIEW']]
filtered_docs = []
for doc in docs:
    tokens = [token.text for token in doc if not token.is_stop]
    filtered_docs.append(tokens)
#bot = docs[10]._.to_bag_of_terms(ngrams=(1, 2, 3), entities=True, weighting="count", as_strings=True)
ngrams = textacy.extract.basics.ngrams(docs[10], 2)
for i in ngrams:
    print(next(ngrams))
df['FILTERED'] = filtered_docs    

words = corpora.Dictionary(filtered_docs)

corpus = [words.doc2bow(doc) for doc in filtered_docs]

LDA = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                      id2word = words,
                                      num_topics = 2,
                                      random_state=2,
                                      update_every=1,
                                      passes=10,
                                      alpha='auto',
                                      per_word_topics=True)
LDA.print_topics()




