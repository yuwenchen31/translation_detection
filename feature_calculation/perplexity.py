#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:13:05 2020

@author: chenfish
"""

#kenLM

import kenlm 
import pandas as pd
import os

data_path = '/Users/chenfish/Desktop/Thesis/Project/data/ht_pe/'

for i in os.listdir(data_path): 
    
    
    if i[-2:] == 'en':

        train = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
    else:
        print('Skip the file.', i)
        continue

    model = kenlm.LanguageModel('/Volumes/Extreme SSD/40m_training_data.binary')
    print('Successfully loaded the kenLM.')
    
    per_sent = []
    for line in train['TRAIN']:
        #line = line.lower()
        #print(line)
        per=model.perplexity(line)
        per_sent.append(per)
        
    train['PER'] = per_sent
    
    train.to_pickle(data_path + i)    