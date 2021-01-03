#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:29:37 2020

@author: chenfish
"""

#news.2018.en.shuffled.deduped

import random
import bz2

data = []

with open('./news.2019.en.shuffled.deduped', 'rb') as f:
    for line in f.readlines():
        data.append(str(line, 'utf-8'))
        
shorten_data = random.sample(data,2000000)  



# for sentence in shorten_data:
#     sentence = sentence.strip()
#     sentence = ' '.join(nltk.word_tokenize(sentence)).lower()
    
    
# with open('shorten_news_crawl_2018','rb') as f:
#     f.write(shorten_data)


# with bz2.open('test_news_30m.bz2','wb') as f:
#     for line in data[:10]:
#         f.write(bytes(line, 'utf-8'))
#     f.close()


with bz2.open('10m_news_crawl.bz2','wb') as f:
    for line in shorten_data:
        f.write(bytes(line, 'utf-8'))
    f.close()












