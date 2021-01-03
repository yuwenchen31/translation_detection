#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:31:47 2020

@author: chenfish
"""
import pandas as pd

src_dir = '../data/mt_ht/raw_data/deepl_source/'
mt_dir = '../data/mt_ht/raw_data/deepl_mt/'
ht_dir = '../data/mt_ht/raw_data/deepl_ht/'

lang_pair = ['deen']

for i in lang_pair: 
    
    print('We are working on ', i)
    
    #txt file is a list. Each sentence is an item. 
    
    
    all_mt= [] 
    with open(mt_dir + i, 'rb') as f:
        for line in f.readlines():
               if not line.strip():
                   continue 
               else:
                    line = line.strip()
                    all_mt.append(str(line, 'utf-8'))
        
    
                
    #read the source file          
    all_src= [] 
    with open(src_dir + i, 'rb') as f:
        for line in f.readlines():
               if not line.strip():
                   continue 
               else:
                    line = line.strip()
                    all_src.append(str(line, 'utf-8'))
                
    
    #get the HT part 
    with open(ht_dir + i ,'r') as f: 
        ht_lines = f.readlines()
        
        temp_ht = []
        all_ht = []
        
        for line in ht_lines:
            
            if line != '\n':
                temp_ht.append(line.rstrip())
                
            else: 
                all_ht.append(temp_ht)
                temp_ht = []
    
    
    all_train = all_mt + all_ht
    all_label = ['MT'] * len(all_mt) + ['HT'] * len(all_ht)
    
    
    try:
        #mt_df = pd.DataFrame({'TRAIN':all_mt, 'LABEL':['MT']*len(all_doc), 'SRC':all_src})
        deepl_df = pd.DataFrame({'TRAIN':all_train, 'LABEL':all_label, 'SRC':all_src*2})
        print('len of MT:', len(all_mt))
        print('len of SRC: ', len(all_src))
        print('len of HT: ', len(all_ht))
        
    except ValueError:
        
        for m, h in zip(all_mt, all_ht):
            if all_mt.index(m) != all_ht.index(h):
                print(m)
                print(h)
                print(all_mt.index(m))
                print(all_ht.index(h))
                
        print('Following sizes do not match....')
        print('len of weird MT:', len(all_mt))
        mis_mt = all_mt
        print('len of weird SRC: ', len(all_src))
        print('len of weird HT: ', len(all_ht))
        mis_src = all_src
        
    
    #deepl_df = all_ht.append(mt_df,ignore_index=True)
    
    
    deepl_df.to_pickle('../data/mt_ht/subset_deepl/train/'+ i)

