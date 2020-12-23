#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:48:51 2020

@author: chenfish
"""

# N = 5, 10, 50
# most frequent words: # of MFW in the chunk / # of tokens in the chunk
# exclude punctuations

import pandas as pd
import string
import os
from wordfreq import top_n_list

data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)

for i in os.listdir(data_path): 
    
    if i[-2:] == 'en':

        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('en', 50)
        
        
    elif i[-2:] == 'de':

        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('de', 10)
                    
    elif i[-2:] == 'ru': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('ru', 10)
        
        
    elif i[-2:] == 'fr': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('fr', 10)
        
        
    elif i[-2:] == 'nl': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('nl', 10)
        
        
    elif i[-2:] == 'pt': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('pt', 10)        
        
    elif i[-2:] == 'fi': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('fi', 10)   
        
    else:
        print('Skip the file.', i)
        continue

    train_data = data['TOK']
    
    all_count = []
    
    
    for doc in train_data:
        
        count = 0
        
        #remove punctuation, lower the word
        flatten_tok = [item.lower() for sublist in doc for item in sublist if item not in string.punctuation]
        
        #print(flatten_tok)
        
        for word in flatten_tok: 
            
            #top_rank is a list
            
            if word in top_rank:
                
                count += 1
                #rank_num += (top_rank.index(word) + 1)
                #print('the word:', word, 'has a rank of',rank_num)
                
                #print(rank_num)
            
            else:
                
                continue
               
                    
                    
        #print('the rank num is:',rank_num)
        #print(len(flatten_tok))
        
        #divide the rank_num by total number of words 
        all_count.append(count/len(flatten_tok))
        
        
    #append this result to the dataframe as RANK
    data['freq 50'] = all_count
    data.to_pickle(data_path + i)
