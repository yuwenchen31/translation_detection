#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:44:15 2020

@author: chenfish
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
import os


# TODO: make the split ratio as 70-15-15!! for (lt,kk,gu)-en, en-gu, mt-pe
# =============================================================================
# # split ratio: 60-20-20
# train, test = train_test_split(file, test_size = 0.2, stratify=file['LABEL'], random_state=42, shuffle=True)
# 
# # dev = 0.8 * 0.25 = 0.2
# train, dev = train_test_split(train, test_size = 0.25, stratify=train['LABEL'], random_state=42, shuffle=True)
# 
# 
# # wmt19: 50-50 for dev and test
# #dev, test = train_test_split(file, test_size = 0.5, stratify=file['LABEL'], random_state=42, shuffle=True)
# =============================================================================


# sequential split for language pair which only has 2019 

# ratio: train-dev-test -> 70-15-15 (lt,kk,gu-en and en-lt)
# ratio: dev-test -> 50-50 (de,fi,ru,zh-en, en-de,ru,fi)


def train_dev_test(file):
    
        # to get balanced labels for both classes from all data, first need to get data from both calsses 
    
    label_name = list(set(file['LABEL']))
    
    # dataframe for both classes
    label_one_df = file[file['LABEL'] == label_name[0]]
    label_two_df = file[file['LABEL'] == label_name[1]]
    print(label_one_df)
    print(label_two_df)
    
    # split point for dataframes of label one: 70-15-15
    first_split = int(len(label_one_df) * 0.7)
    second_split = int(len(label_one_df) * 0.85)
    
    
    print('first split:', first_split)
    print('second split:', second_split)
    
    
    #final train, dev, test is the sum of the split in each dataframe
    #train: [:70%]
    train = pd.concat([label_one_df[:first_split],label_two_df[:first_split]], ignore_index=True)
    
    #dev: [70%:85%]
    dev = pd.concat([label_one_df[first_split:second_split],label_two_df[first_split:second_split]], ignore_index=True)
    
    #test: [85%:]
    test = pd.concat([label_one_df[second_split:],label_two_df[second_split:]], ignore_index=True)
    
    
    print('train length:', len(train))
    print(label_name[0], 'in train has', len(train[train['LABEL']==label_name[0]]))
    print(label_name[1], 'in train has', len(train[train['LABEL']==label_name[1]]))
    
    print('dev length:', len(dev))
    print(label_name[0], 'in dev has', len(dev[dev['LABEL']==label_name[0]]))
    print(label_name[1], 'in dev has', len(dev[dev['LABEL']==label_name[1]]))
    
    
    print('test length:', len(test))
    print(label_name[0], 'in test has', len(test[test['LABEL']==label_name[0]]))
    print(label_name[1], 'in test has', len(test[test['LABEL']==label_name[1]]))
    
    
    return train, dev, test
    

    
def dev_test(file):
    
    # to get balanced labels for both classes from all data, first need to get data from both calsses 
    
    label_name = list(set(file['LABEL']))
    
    
    # dataframe for both classes
    label_one_df = file[file['LABEL'] == label_name[0]]
    label_two_df = file[file['LABEL'] == label_name[1]]
    
    print(label_one_df)
    print(label_two_df)
    # split point for dataframes of label one: 50-50
    first_split = int(len(label_one_df) * 0.5)


    print('first split:', first_split)


    #final dev, test
    
    # dev: label_one[:50%] + label_two[:50%]
    dev = pd.concat([label_one_df[:first_split],label_two_df[:first_split]], ignore_index=True)
    
    #test: [50%:]
    test = pd.concat([label_one_df[first_split:],label_two_df[first_split:]], ignore_index=True)
    
    
    print('dev length:', len(dev))
    print(label_name[0], 'in dev has', len(dev[dev['LABEL']==label_name[0]]))
    print(label_name[1], 'in dev has', len(dev[dev['LABEL']==label_name[1]]))
    
    
    print('test length:', len(test))
    print(label_name[0], 'in test has', len(test[test['LABEL']==label_name[0]]))
    print(label_name[1], 'in test has', len(test[test['LABEL']==label_name[1]]))
    
    dev_count = 0 
    for i in dev['TRAIN']:
        dev_count += len(i)
        
    print('dev count:', dev_count)
    
    
    test_count = 0 
    for i in dev['TRAIN']:
        test_count += len(i)
        
    print('test_count',test_count)
    
    return dev, test
    


if __name__ == "__main__":


    data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
    
    for d in os.listdir(data_path): 
    #for d in ['2.ende','5.ende','10.ende']: 
        try: 
        
            if not d.startswith('.'):
        
                data = pd.read_pickle(data_path + d)
                print('Now we are working on', d)
                
            else:
                print('Skip the dot file.')
                continue
            
        except IsADirectoryError:
            continue
        
        train,dev,test = train_dev_test(data)
        #dev, test = dev_test(data)
        
        train.to_pickle("/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/" + 'train.' + d)
        dev.to_pickle("/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/" + 'dev.' + d)
        test.to_pickle("/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/" + 'test.' + d)
    
