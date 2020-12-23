#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:50:04 2020

@author: chenfish
"""

# This file is to calculate the average perplexity of a subset of sentences
# and append a column 'PER' to the orginal dataframe
# NOTE: so far it only had english language model 


# compute the average perplexity 
# for document level, subset sentence level 

import kenlm 
import pandas as pd
import os


data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
model = kenlm.LanguageModel('/Volumes/YuWen/40m_training_data.binary')
print('Successfully loaded the kenLM.')

for i in os.listdir(data_path): 
    
    
    if i[-2:] == 'en':

        train = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
    else:
        print('Skip the file.', i)
        continue
    
    data = train['TRAIN']
    
    per_sent = []
    
    for doc in data:
        for line in doc: 
        #line = line.lower()
        #print(line)
            per=model.perplexity(line)
            per += per
        
        final_per = per/len(doc)
        per_sent.append(round(final_per,4))
        
    train['PER'] = per_sent
    
    #train.to_pickle('./Data/playground/' + i)
    
    train.to_pickle(data_path + i)    