# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:26:16 2022

@author: starg
"""

test = 'this is a super duper cool test sentence that we can use in the freqdist so that we can see how often some of the words being said are being said'

import nltk
import matplotlib.pyplot as plt

wt = nltk.tokenize.word_tokenize(test)

fdist = nltk.probability.FreqDist(wt)
fdist.plot(10)
print(fdist.items())

fdist = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True) )

print(fdist.items())

plt.plot(fdist.keys(),fdist.values(), 'r-')
plt.show()