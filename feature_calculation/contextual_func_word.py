#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:14:16 2020

@author: chenfish
"""

# contextual function words 
# # of trigrams consisting of at least 2 function words / # of trigrams 


import pandas as pd
import nltk
import math
from builtins import sum
import os
from collections import Counter


data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)
print('We are working on contextual function words.')


for d in os.listdir(data_path): 
    
    
    if not d.startswith('.'):
        try: 
            file = pd.read_pickle(data_path + d)
            print('Now we are working on', d)
        
        except IsADirectoryError:
            print('Skip the file.', d)
            continue
        
    else:
        print('Skip the file.', d)
        continue


    text = file['UD']
    
    #flatten the whole corpus 
    #t = [word for doc in text for sent in doc for word in sent]
    
    
    #trigram_freq = nltk.FreqDist(nltk.trigrams(t))
    
    context_func = []
    

    
    for doc in text: 
        
        count = 0
        trigram_count = 0 
    
    
        for line in doc:
            
            trigram_dict = nltk.FreqDist(nltk.trigrams(line))
            trigram_count += len(trigram_dict)
            
            for i in trigram_dict.keys():
                a = Counter(i)
                n = sum(a[x] for x in ['aux','cop','mark','det','clf','case'])
                if n >= 2:
                    count += 1
           
                
        try: 
            context_func.append(round(count/trigram_count,4))
            
        except ZeroDivisionError: 
            context_func.append(0)


    #append this result to the dataframe as RANK
    file['CONTEXT FUNC'] = context_func
    
    print('saving the file...', d)
    file.to_pickle(data_path + d)


