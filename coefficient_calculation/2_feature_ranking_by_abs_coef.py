#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:52:50 2020

@author: yuwen
"""

# This file implements the features ranking based on absolute coefficients 
# input files: ../coefficient_table/context_length_all.csv
# output files: ../coefficient_table/ranking_[...].csv 

import statistics
import pandas as pd 

# for mtht & mtpe
file = pd.read_csv("/data/s3619362/coefficient_table/context_length_all_mtht.csv")


# get the absolute value of coefficient 
file['Absolute coefficient'] = [round(abs(coef),3) for coef in file['Coefficient']]


new_file = pd.DataFrame()


# for mtht & mtpe
for i in file.groupby(['Language Pair', 'Context Length']):
    
# for htpe-zhen (&mtht-zhen)
#for i in file.groupby(['Context Length']):
    temp_df = i[1].sort_values(by=['Absolute coefficient'],ascending=False)
    temp_df['Rank'] = range(1,len(temp_df)+1)
    new_file = new_file.append(temp_df)
    

new_file.to_csv("/data/s3619362/coefficient_table/ranking_mtht.csv")
    

