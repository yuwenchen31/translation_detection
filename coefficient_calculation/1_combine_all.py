#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:13:33 2020

@author: yuwen
"""
# This file combines different language pairs and context length into one csv file. 
# input files: csv files of different lang pairs in different context length
# output file: ../coefficient_table/context_length_all.csv

import pandas as pd 
import os


context_length = [2,5,10]

data_path = '/data/s3619362/selected_features/'
save_path = '/data/s3619362/coefficient_table/'
    
final = []

# for features before VIF filter 
# coef_df = pd.DataFrame(columns=['Context Length','Feature Name', 'Language Pair', 'Coefficient'])

# for features after VIF filter 
coef_df = pd.DataFrame(columns=['Context Length','Feature', 'Language Pair', 'Coefficient','VIF'])


for c in context_length:
    
    #separate different context length into different files
    #coef_df = pd.DataFrame(columns=['Context Length','Feature Name', 'Language Pair', 'Coefficient'])
    
    for d in os.listdir(data_path): 
    
        if not d.startswith('.'):
            
            try: 
        
                orig_file = pd.read_csv(data_path + d)
            
                
                temp_df = orig_file[orig_file['Context Length']==c]
                
                # for mtht, mtpe
                temp_df['Language Pair'] = d[:4]

                coef_df = coef_df.append(temp_df, ignore_index=True)
                
                #separate different context length into different files
                #coef_df.to_csv(save_path + 'context_length_' + str(c) + '.csv')
                
            except IsADirectoryError:
                print('Skip the file.', d)
                continue

        
        else:
            print('Skip the file.', d)
            continue
        
# for all context length & lang pairs into one file
coef_df.to_csv(save_path + 'context_length_all.csv')


    