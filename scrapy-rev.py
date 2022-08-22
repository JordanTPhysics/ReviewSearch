# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:01:57 2022

@author: starg
"""

import scrapy

class ReviewSpider(scrapy.spider):
    
    name = 'review'
    start_urls = []
    
    def parse(self, response):
        
        for review in response.css(''):
        
            yield {
                'text': '',
                'likes': '',
                'date': '',
                'rating': ''
            }