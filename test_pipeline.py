# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:21:06 2022

@author: starg
"""
import re
import string
import unittest


def clean_text(text):
    string_punctuation = string.punctuation + "``“”£"

    # lowercase
    # not needed???

    # replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    # remove instances of punctuation
    text = re.sub('[%s]' % re.escape(string_punctuation), '', text)
    # remove numbers and words attached to numbers
    text = re.sub('\w*\d\w*', '', text)

    return text


class TestPipeline(unittest.TestCase):

    def test_clean_text_works(self):
        # Arrange
        test_text = """
        The purpose of this text...
        is to ensure that the data cleaning fucntions remove punctuation like this:
        !/? and digits like this: 67567 or words containing digits like this tree4534
        """
        expected_text = """
        The purpose of this text
        is to ensure that the data cleaning fucntions remove punctuation like this
         and digits like this  or words containing digits like this 
        """
        # Act
        result = clean_text(test_text)
        # Assert
        self.assertEqual(result, expected_text)
