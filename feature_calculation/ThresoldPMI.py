#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:50:00 2020

@author: chenfish
"""

#Threshold PMI: # of bigram PMI > 0 / # of all bigram


import pandas as pd
import nltk
import math
from builtins import sum
import os


data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)
print("We are now working thresold pmi.")


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
    
    
    normalized_count = []
    
    for doc in text: 
    
        pmi = 0    
        count = 0
    
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
          #print(bigram_pmi)
          
          if bigram_pmi >0: 
              count += 1
          else: 
              #print('bigram PMI is < 0')
              continue
    
        normalized_count.append(count/len(bigram_dict))



    #append this result to the dataframe as RANK
    data['ThresoldPMI'] = normalized_count
    
    print("saving the file...")
    data.to_pickle(data_path + d)


