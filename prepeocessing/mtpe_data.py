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


#read the mt file
mt = []
with open('/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/raw_data/enru.dev.mt', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        mt.append(str(line, 'utf-8'))
        
        
#read the pe file
pe = []
with open('/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/raw_data/enru.dev.pe', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        pe.append(str(line, 'utf-8'))
        
        
        
#read the src file
src = []
with open('/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/raw_data/enru.dev.src', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        src.append(str(line, 'utf-8'))        

#make them a subset of 10 sentences 
mt_sub = subset_sentences(mt,num=10)
pe_sub = subset_sentences(pe,num=10)        
src_sub = subset_sentences(src,num=10)

#make the training data and labels 
training_data = mt_sub + pe_sub
true_label = ['MT'] * len(mt_sub) + ['PE'] * len(pe_sub)


#create datafream has three columns: 'TRAIN' and 'LABEL' and 'SRC'
df = pd.DataFrame(data={"TRAIN": training_data, "LABEL": true_label, "SRC":src_sub*2})


#save this to ht-pe directory 
df.to_pickle('/Users/chenfish/Desktop/Thesis/Project/data/mt_pe/all/10/enru.dev')
