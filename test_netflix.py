# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:18:43 2021

@author: starg
"""

import unittest
from SpaCy import space1
import string

testText = """The purpose of this text...
 is to ensure that the data cleaning fucntions remove punctuation like this:
    !/? and digits like this: 67567 or words containing digits like this tree4534
"""

class TestData(unittest.TestCase):
    def test_textclean(self):   ##check if text was cleaned
        self.assertNotRegex(space1.clean_text_round1(testText), string.punctuation)
    
    
    if __name__ == '__main__':
        unittest.main()