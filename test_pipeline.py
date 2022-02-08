# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:21:06 2022

@author: starg
"""
import nltk
import unittest
from Pipeline import clean_text_round1

testText = """The purpose of this text...
 is to ensure that the data cleaning fucntions remove punctuation like this:
    !/? and digits like this: 67567 or words containing digits like this tree4534
"""

class TestPipeline(unittest.TestCase):
    def test_clean(self):
        self.assertNotEquals(clean_text_round1(testText), testText)
        