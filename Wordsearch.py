# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:58:30 2021

@author: kapo
"""

import nltk

from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
reviewCamel = "For the first time ever a launch day console, something I have been buying since the PS1, has failed me. HDMI port is clearly broken and all hdmi cables feel wobbly in it. Worked for 45 minutes, then failed to register. Worked again for another 30 minutes then total failure after that. " \
         "Nothing will remedy this. Sony claim 15 working days to fix and if they can't then they will replace it 'stock depending'. " \
         "And we all know how much stock there is at the moment. I dont think I could be more disappointed, really unhappy with Sony right about now."
review = reviewCamel.lower()

#tokenized = word_tokenize(review)
lemmatizer = WordNetLemmatizer()
filtered_text = []
tokenizer = RegexpTokenizer(r'\w+')
reviewCleaned = tokenizer.tokenize(review)
for each_word in reviewCleaned:
    if each_word not in stop_words:
        filtered_text.append(each_word)#print('Tokenized no stop words: {}' .format(filtered_text))

tagged = nltk.pos_tag(reviewCleaned)
for (each_word, tag) in tagged:
    if tag == 'NN':
        filtered_text.append(each_word)
        
print(filtered_text)
        
        
        



print('Tokenized no stop words:{}' .format(filtered_text))

