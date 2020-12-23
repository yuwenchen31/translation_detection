#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:52:16 2020

@author: chenfish
"""

#position token frequency 

import math
from builtins import sum
import pandas as pd 
from collections import Counter
import os


data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'

print(data_path)
print('We are working on positional token frequency.')


for d in os.listdir(data_path): 
    
    
    if not d.startswith('.'):
        try: 
            data = pd.read_pickle(data_path + d)
            print('Now we are working on', d)
        
        except IsADirectoryError:
            print('Skip the file.', d)
            continue
        
    else:
        print('Skip the file.', d)
        continue


    # 1. take 1st, 2nd, 3rd last, 2nd last, last word from from all sentences. Make a dictionary of it. 

    
    text = data['TOK']
    word_count = dict()
    
    
    count_1 = Counter((sent[0].lower() for doc in text for sent in doc))
    count_2 = Counter((sent[1].lower() for doc in text for sent in doc if len(sent) > 2))
    count_last = Counter((sent[-1].lower() for doc in text for sent in doc))
    count_2last = Counter((sent[-2].lower() for doc in text for sent in doc if len(sent) > 2))
    count_3last = Counter((sent[-3].lower() for doc in text for sent in doc if len(sent) > 3))
    
    
    all_count = count_1 + count_2 + count_last + count_2last + count_3last
    
    
    # 2. make a dic of unique word's frequency : dic[word]/sum(dic.values)
    
    word_freq = dict()
    
    sum_all_count = sum(all_count.values())
    
    for key, values in all_count.items():
        word_freq[key] = values/sum_all_count
    
    
    # 3. for each training sentence, get the frequency of 1st, 2nd, 3rd last, 2nd last, last word, sum them
    
    sum_freq = []
    
    for doc in text:
        sum_sent_freq = 0 
        
        for sent in doc: 
            
            try: 
                sent_freq = word_freq[sent[0].lower()] + word_freq[sent[1].lower()] + word_freq[sent[-1].lower()] + word_freq[sent[-2].lower()] + word_freq[sent[-3].lower()]
            
                sum_sent_freq += sent_freq
                
            except IndexError:
                
                sum_sent_freq = 0 
        
        sum_freq.append(round(sum_sent_freq,4))
        
    #append this result to the dataframe as PTF
    data['PTF'] = sum_freq
    
    data.to_pickle(data_path + d)






