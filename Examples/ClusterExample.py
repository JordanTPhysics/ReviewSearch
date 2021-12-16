# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:30:59 2021

@author: starg

"""
import pprint as pp
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.book import *
from nltk.corpus import gutenberg
from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance
import matplotlib.pyplot as plt


emma_vec = Word2Vec(gutenberg.sents('austen-emma.txt'))
leaves_vec = Word2Vec(gutenberg.sents('whitman-leaves.txt'))
#print(emma_vec.wv.most_similar('pain', topn=6))
#print(leaves_vec.wv.most_similar('pain', topn=6))

vocab = nltk.FreqDist(text1)
#print(len(vocab))
#print(vocab.most_common(20))


w = wn.synsets("unmitigated")[0]
#print(w.name(), '-', w.definition())
#print(w.examples())


print(punctuation)
without_punct = [w for w in text1 if w not in punctuation]  # this is called a list comprehension


sw = stopwords.words('english')
print(sw)
without_sw = [w for w in without_punct if w not in sw] 

#print(len(text1))
#print(len(without_punct))
#print(len(without_sw))


st = PorterStemmer()
words = ['is', 'are', 'bought', 'buys', 'giving', 'jumps', 'jumped', 'birds', 'do', 'does', 'did', 'doing']
for word in words:
    print(word, st.stem(word))
    
    


bible_sents = gutenberg.sents('bible-kjv.txt')
sw = stopwords.words('english')
bible = [[w.lower() for w in s if w not in punctuation and w not in sw] for s in bible_sents]
print(len(bible))

bible_vec = Word2Vec(bible)
pp.pprint(bible_vec.wv.most_similar('god', topn=8))
pp.pprint(bible_vec.wv.most_similar('creation', topn=5))



vectors = [np.array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
means = [[4, 3], [5, 5]]

clusterer = KMeansClusterer(2, euclidean_distance, initial_means=means)
clusters = clusterer.cluster(vectors, True, trace=True)

print('Clustered:', vectors)
print('As:', clusters)
print('Means:', clusterer.means())

vectors = [np.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
clusterer = KMeansClusterer(2, euclidean_distance, repeats=10)
clusters = clusterer.cluster(vectors, True)
centroids = clusterer.means()
print('Clustered:', vectors)
print('As:', clusters)
print('Means:', centroids)

# classify a new vector
vector = np.array([2,2])
print('classify(%s):' % vector, end=' ')
print(clusterer.classify(vector))

x0 = np.array([x[0] for idx, x in enumerate(vectors) if clusters[idx]==0])
y0 = np.array([x[1] for idx, x in enumerate(vectors) if clusters[idx]==0])
plt.scatter(x0,y0, color='blue')
x1 = np.array([x[0] for idx, x in enumerate(vectors) if clusters[idx]==1])
y1 = np.array([x[1] for idx, x in enumerate(vectors) if clusters[idx]==1])
plt.scatter(x1,y1, color='red')

xc = np.array([x[0] for x in centroids])
yc = np.array([x[1] for x in centroids])
plt.scatter(xc,yc, color='orange')
plt.show()