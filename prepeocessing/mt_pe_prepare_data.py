#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:02:15 2020

@author: chenfish
"""
import sys
sys.path.append("../")

from util.utils import get_orig, subset_sentences, corpy_udpipe, make_training_and_labels, get_pos_ud_head
import pandas as pd


# turn the pe & mt files into a df of 3 columns: train, label, src (EN)
# for deepL preparation as well

for lang in ['deen','ende','enru']:

    #read the mt file
    mt = []
    with open('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/deepl_mt/mt_' + lang + '_test.txt', 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            mt.append(str(line, 'utf-8'))
            
            
    #read the pe file
    ht = []
    with open('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/deepl_ht/ht_'+ lang + '_test.txt', 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            ht.append(str(line, 'utf-8'))
            
            
    #read the src file
    src = []
    with open('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/deepl_src/src_'+ lang + '_test.txt', 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            src.append(str(line, 'utf-8'))        
    
    #make them a subset of 10 sentences 
    mt_sub = subset_sentences(mt,num=10)
    ht_sub = subset_sentences(ht,num=10)        
    src_sub = subset_sentences(src,num=10)
    
    #make the training data and labels 
    training_data = mt_sub + ht_sub
    true_label = ['MT'] * len(mt_sub) + ['HT'] * len(ht_sub)
    
    #create datafream has three columns: 'TRAIN' and 'LABEL' and 'SRC'
    df = pd.DataFrame(data={"TRAIN": training_data, "LABEL": true_label, "SRC":src_sub*2})
    
    #save this to ht-pe directory 
    df.to_pickle('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/deepl/test.10.' + lang)
