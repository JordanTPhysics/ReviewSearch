# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:51:44 2021
Document term matrix for NLP
@author: starg
"""
import pickle
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
import re
from nltk import word_tokenize, pos_tag
from gensim import matutils, models
import scipy.sparse
index = []
for i in range (259):
    index.append(i)
    i+1

stopList = list(nltk.corpus.stopwords.words('english'))
#add additional stop words relevant to this context
stopList.extend(['Netflix','netflix','account','...','I','And','service','movies','customer','\'d'])
stopList.extend(punctuation)
df = pd.read_csv(r'reviewdata.csv',header=[0])
df.set_index(pd.Index(index),'ID')
df.columns=['DATE','REVIEW','RATING']
df.dropna(subset =["DATE","REVIEW"],inplace=True)

filtered_reviews = []
reviewlist = list(df.REVIEW)
def cleantext(review):
   output = ''.join(c for c in review if not c.isdigit())
   #tokens = wt(output)
   #filtered_review = [w.lower() for w in tokens if w not in stopList and w not in punctuation]
   
   return output

for review in reviewlist:
    filtered_review = cleantext(review)
    filtered_reviews.append(filtered_review)
    

    

del(df['REVIEW'])

df['Filtered'] = filtered_reviews
vect = CountVectorizer()
DTMvectors = vect.fit_transform(df.Filtered)

td = pd.DataFrame(DTMvectors.todense()).iloc[:259]
#display the given word for each column
td.columns = vect.get_feature_names()
#transpose
tdm = td.T
#display document ID for each
tdm.columns = ['Doc '+str(i) for i in range(1, 200)]
tdm['total_count'] = tdm.sum(axis=1)

# Top 25 words 
tdm = tdm.sort_values(by ='total_count',ascending=False)[:25] 

# Print the first 10 rows 
print(tdm.drop(columns=['total_count']).head(10))


sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()



def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)