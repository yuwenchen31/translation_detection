#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:27:09 2020

@author: chenfish
"""


# pos perplexity 

import kenlm 
import pandas as pd
import os


source_model = kenlm.LanguageModel('/Users/yuwen/Desktop/Thesis/Project/data/language_model_2m/zh.binary')
target_model = kenlm.LanguageModel('/Users/yuwen/Desktop/Thesis/Project/data/language_model_2m/en.binary')

file_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/mtht_10_zhen'
print(file_path)
print('We are now working on perplexity difference.')

file = pd.read_pickle(file_path)

data = file['POS']

per_diff = []

for doc in data:
    
    source_sum = 0
    target_sum = 0
    
    for line in doc: 
        
        # convert list into string
        line = ' '.join(line)
        line = line.lower()
        #print(line)
        
        source_per_sent = source_model.perplexity(line)
        #print(source_per_sent)
        
        source_sum = source_sum + source_per_sent
        #print('sum=', source_sum)
        
        
        target_per_sent = target_model.perplexity(line.lower())
        #print(target_per_sent)
        target_sum = target_sum + target_per_sent
        #print('sum= ', target_sum)
        
        
    doc_source_per = source_sum/len(doc)
    #print(doc_source_per)
    doc_target_per = target_sum/len(doc)
    #print(doc_target_per)
    
    doc_per_diff = doc_source_per - doc_target_per
    
    per_diff.append(doc_per_diff)
    

file['PER diff'] = per_diff
file.to_pickle(file_path)
