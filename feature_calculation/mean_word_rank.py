#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:27:48 2020

@author: chenfish
"""
import pandas as pd
import string
import os
from wordfreq import top_n_list



#data_path = '/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/dev/'

data_path =  '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)
print('We are working on 5000 word rank.')

for i in os.listdir(data_path): 
    
    if i[-2:] == 'en':

        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('en', 5000)
        
        
    elif i[-2:] == 'de':

        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('de', 5000)
                    
    elif i[-2:] == 'ru': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
        
        top_rank = top_n_list('ru', 5000)
        
        
    elif i[-2:] == 'fr': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('fr', 5000)
        
        
    elif i[-2:] == 'nl': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('nl', 5000)
        
        
    elif i[-2:] == 'pt': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('pt', 5000)    
        
        
    elif i[-2:] == 'fi': 
        data = pd.read_pickle(data_path + i)
        print('Now we are working on', i)
    
        top_rank = top_n_list('fi', 5000)    
        
        
    else:
        print('Skip the file.', i)
        continue

    train_data = data['TOK']
    
    final_rank = []
    
    
    for doc in train_data:
        
        rank_num = 0
        
        #remove punctuation, lower the word
        flatten_tok = [item.lower() for sublist in doc for item in sublist if item not in string.punctuation]
        
        #print(flatten_tok)
        
        for word in flatten_tok: 
            
            #top_rank is a list
            
            if word in top_rank:
                
                rank_num += (top_rank.index(word) + 1)
                #print('the word:', word, 'has a rank of',rank_num)
                
                #print(rank_num)
            
            else:
                
                #print('This word is not in the list or it is a punct. Give this word a rank of 6000.', word)
                rank_num += 6000
                #print(rank_num)
                    
                    
        #print('the rank num is:',rank_num)
        #print(len(flatten_tok))
        
        #divide the rank_num by total number of words 
        final_rank.append(rank_num/len(flatten_tok))
        
        
    #append this result to the dataframe as RANK
    data['RANK'] = final_rank   
    data.to_pickle(data_path + i)