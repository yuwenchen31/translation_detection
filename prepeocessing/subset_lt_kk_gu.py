#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:21:10 2020

@author: chenfish
"""

# prepare data for lt, kk, gu > en
# make them a subset of 10 sentences 

# train test split into train/dev/test files 

import sys
sys.path.append("../")

from util.utils import read_corpus, make_training_and_labels, generate_file_name, subset_sentences
import pandas as pd



lang_pair = ['zhen']


for i in lang_pair:
    
    print('Now we are processing', '2017', i)
    
    mt_file, ht_file, src_file = generate_file_name('2017', i, sgm=False)
    print('MT filename is: ', mt_file)
    print('HT filename is: ', ht_file)
    print('SRC filename is: ', src_file)
    
    mt = read_corpus(mt_file)
    ht = read_corpus(ht_file)
    src = read_corpus(src_file)
    
    
    #turn into a subset of 10 sentences 
    mt_sub = subset_sentences(mt, num=10)
    ht_sub = subset_sentences(ht, num=10)
    src_sub = subset_sentences(src, num=10)
    
    print('mt # at subset level:', len(mt_sub))
    print('ht # at subset level:', len(ht_sub))
    print('src # at subset level:', len(src_sub))
    
    training_data, true_label = make_training_and_labels(mt_sub, ht_sub)
    
    df = pd.DataFrame(data={"TRAIN": training_data, "LABEL": true_label, "SRC":src_sub*2})
    
    df.to_pickle('/Users/chenfish/Desktop/Thesis/Project/data/'+ str(i) + '2017')
    