#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:54:38 2020

@author: chenfish
"""

import pandas as pd
import nltk
import math
from builtins import sum
import os


#data_path = '/Users/chenfish/Desktop/Thesis/Project/data/mt_ht/subset_sent/ltkkgu/'
data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)
print('We are working on PMI.')

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

    text = data['TOK']
    
    #flatten the whole corpus 
    t = [word for doc in text for sent in doc for word in sent]
    
    
    unigram_freq = nltk.FreqDist(t)
    bigram_freq = nltk.FreqDist(nltk.bigrams(t))
    
    ave_pmi = []
    
    for doc in text: 
    
        pmi = 0    
    
        #flatten each doc 
        flatten_doc = [word for sent in doc for word in sent]
        
        #get the bigram dict of flatten doc
        bigram_dict = nltk.FreqDist(nltk.bigrams(flatten_doc))
        
        
        for i in bigram_dict.keys():
          prob_word1 = unigram_freq[i[0]] / float(sum(unigram_freq.values()))
          #print(prob_word1)
          prob_word2 = unigram_freq[i[1]] / float(sum(unigram_freq.values()))
          #print(prob_word2)
          prob_word1_word2 = bigram_freq[(i[0],i[1])] / float(sum(bigram_freq.values()))
          #print(prob_word1_word2)
          bigram_pmi = math.log(prob_word1_word2/float(prob_word1*prob_word2),2)
          
          pmi += bigram_pmi
    
    
        ave_pmi.append(pmi/len(bigram_dict))
        #print(ave_pmi[:5])
        

    
    #append this result to the dataframe as RANK
    data['PMI'] = ave_pmi
    
    data.to_pickle(data_path + d)


