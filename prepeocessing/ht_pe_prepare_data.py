#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:29:05 2020

@author: chenfish
"""

import sys
sys.path.append("../")

from util.utils import get_orig, subset_sentences, corpy_udpipe, make_training_and_labels, get_pos_ud_head
import pandas as pd


# modify the MS files as pickple file with 10 columns: TRAIN, LABEL(HT & PE), SRC, POS, UD, HEAD, TOK, PER, RANK, PMI


#get the non-translationese sentences (list of strings) 
_,non_trans_ht, src = get_orig('newstest2017.SogouKnowingnmt.5171.zhen.sgm', 'newstest2017-zhen-ref.en.sgm', 'newstest2017-zhen-src.zh.sgm', 'zh')


#get the official WMT reference
wmt_ht = []
with open('/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/raw_data/newstest2017-zhen-ref.en', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        wmt_ht.append(str(line, 'utf-8'))


#get the index of non-translationese part 
ind_non_trans = [wmt_ht.index(sent) for sent in wmt_ht if sent in non_trans_ht]


ht = []
pe = []
mt = []

#read the file: REFERENCE-HT
with open('/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/raw_data/Translator-HumanParityData-Reference-HT.txt', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        ht.append(str(line, 'utf-8'))


#read the file: REFERENCE-PE
# with open('/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/raw_data/Translator-HumanParityData-Reference-PE.txt', 'rb') as f:
#     for line in f.readlines():
#         line = line.strip()
#         pe.append(str(line, 'utf-8'))
        
#read the file: Microsoft-MT(combo6)
with open('/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/raw_data/Translator-HumanParityData-Combo-6.txt', 'rb') as f:
    for line in f.readlines():
        line = line.strip()
        mt.append(str(line, 'utf-8'))
    

# get the non-translationese part of ht & pe(mt) using ind_non_trans
ht = [ht[i] for i in ind_non_trans]
#pe = [pe[i] for i in ind_non_trans]
mt = [mt[i] for i in ind_non_trans]

#make them a subset of 4 sentences 
ht_sub = subset_sentences(ht, num=10)
#pe_sub = subset_sentences(pe, num=2)
src_sub = subset_sentences(src, num=10)
mt_sub = subset_sentences(mt, num=10)

#make the training data and labels 
#training_data = ht_sub + pe_sub
training_data = ht_sub + mt_sub
true_label = ['HT'] * len(ht_sub) + ['MT'] * len(mt_sub)


#create datafream has three columns: 'TRAIN' and 'LABEL' and 'SRC'
df = pd.DataFrame(data={"TRAIN": training_data, "LABEL": true_label, "SRC":src_sub*2})


#save this to ht-pe directory 
df.to_pickle('/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht_10_ms')




