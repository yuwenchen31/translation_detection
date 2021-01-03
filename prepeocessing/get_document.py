#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:19:58 2020

@author: chenfish
"""

import sys
sys.path.append("../")

from util.utils import get_orig, make_training_and_labels, generate_file_name
import pandas as pd

# This file is to process the raw corpus from WMT 
# and get the non-translationese part (document-level) and turn them into 3 column ('TRAIN', 'LABEL', 'SRC') dataframe


#combine 2017-2019, put them into a pd.pickle, three columns: TRAIN(MT+HT), LABEL, SRC
#for each language pairs

# en - de, fi, ru, zh; de, ru - en: use 2019 data as dev set

#only have 2019: lten, kken, guen, enlt



year = ['2016','2017','2018']

#year = ['2019']
lang_pair = ['deen']


for i in lang_pair:
    df = pd.DataFrame(columns = ['TRAIN', 'LABEL', 'SRC'])
    
    
    for j in year: 
        print('Now we are processing', j, i)
        mt_file, ht_file, src_file = generate_file_name(j, i)
        print(mt_file, ht_file, src_file)
        #mt, ht, src = get_orig(mt_file, ht_file, src_file, i[:2],document_level=False)
        
        
        # to get the source for deepL
        mt, ht, src = get_orig(mt_file, ht_file, src_file, i[:2],document_level=False)
        
        
        # check sentence #
        print('mt doc #:', len(mt))
        print('ht sent #:', len(ht))
        print('src sent #:', len(src))
        

        
        #print('all:', len(mt)+len(ht))
        #training_data, true_label = make_training_and_labels(mt, ht)
        
        #new_df = pd.DataFrame(data={"TRAIN": training_data, "LABEL": true_label, "SRC":src*2})
        
        #df = df.append(new_df,ignore_index=True)
        
        
        # create src file for deepL
        # with open('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/deepl_source/src_' + i + '_test.txt', 'a') as f:
        #     for item in src:
        #         f.write("%s\n" % item)
        
        
        #create ht file for deepL
        with open('/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/deepl_ht/ht_' + i + '.txt', 'a') as f:
            for item in ht:
                f.write("%s\n" % item)
        
        
    #df.to_pickle('./Data/Combined_Doc/'+ str(i))


