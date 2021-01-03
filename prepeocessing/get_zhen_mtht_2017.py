#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:13:31 2020

@author: chenfish
"""

import pandas as pd 

# make the mt-ht zhen train-dev-test split the same as ht-pe

htpe_train = pd.read_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/ht_pe/train.zhen")
htpe_dev = pd.read_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/ht_pe/dev.zhen")
htpe_test = pd.read_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/ht_pe/test.zhen")


htpe_train_index = htpe_train.index.tolist()
htpe_dev_index = htpe_dev.index.tolist()
htpe_test_index = htpe_test.index.tolist()


mtht = pd.read_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/mt_ht/2017zhen")

mtht_train = mtht.loc[htpe_train_index]
mtht_train.to_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/mt_ht/train.zhen.2017")

mtht_dev = mtht.loc[htpe_dev_index]
mtht_dev.to_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/mt_ht/dev.zhen.2017")

mtht_test = mtht.loc[htpe_test_index]
mtht_test.to_pickle("/Users/chenfish/Desktop/Thesis/Project/Data/mt_ht/test.zhen.2017")
