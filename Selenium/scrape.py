# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:54:01 2022

@author: starg
"""

from bs4 import BeautifulSoup
import requests

page = requests.get('https://m.facebook.com/pg/CanterburyRiverTours/reviews')


doc = BeautifulSoup(page.text, 'html.parser')
spans = doc.find_all('span')

dates = doc.find_all(text='about')
print(dates)
for i in spans:
    print(i)
    ps = i.find_all('p')
    print(ps)


