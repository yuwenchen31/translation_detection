#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:37:56 2020

@author: chenfish
"""

# count the syllable per word, normalized by token counts
from big_phoney import BigPhoney
import string 
import pandas as pd
import os 
import re

#initialize the syllable counter 
phoney = BigPhoney()

#data_path = '/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/dev/'

data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'


for i in os.listdir(data_path): 
    
    if i[-2:] == 'en':

        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        
    else: 
        print('Skip the file.', i)
        continue #return to the top of the loop 
    
    
    syllable_ratio = []
    
    text = data['TRAIN']
    
    pattern = r'[^A-Za-z ]'
    regex = re.compile(pattern)
    
    for doc in text:
        
        syllables = 0
        
        #flaten the doc 
        #flatten_doc = [word for sent in doc for word in sent if word not in string.punctuation]
        
        for line in doc: 
            line = regex.sub('', line)
            #get the syllables per line 
            syllables += phoney.count_syllables(line)

        
        syllable_ratio.append(syllables/len(doc))
    

    print(data.shape)
    #append this result to the dataframe as RANK
    data['SYL'] = syllable_ratio
    data.to_pickle(data_path + i)
        
    