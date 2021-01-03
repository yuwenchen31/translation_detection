#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:19:42 2020

@author: yuwen
"""
import statistics
import pandas as pd 

# This file calculates two metrics: 1. average coefficients per feature 2. feature ranking based on absolute coefficients  
# input file: ../coefficient_table/ranking_[...].csv
# output file: 1. coef_table 2. rank_table in ../coefficient_table/



# read csv file *derived from feature_ranking_by_abs_coef.py)
file = pd.read_csv('/data/s3619362//all_feature_with_ranking_mtht.csv')

    
def average_coef_rank(file, rank = True):

    # add the value of the same features 
    
    # if we only want the table for context length of 10
    #file = file[file['Context Length'] == 10]
    
    feature_name = set(file['Feature'])
    all_lang = set(file['Language Pair'])
    print(len(all_lang))

    final_feat = []
    final_value = []
    final_std = []
    final_appear =[]
    
    for f in feature_name:

        print(f)
        
        if rank:
        
            sum_value = file.loc[file['Feature'] == f, 'Rank'].sum()
            std = round(file.loc[file['Feature'] == f, 'Rank'].std(),2)
            num_pair = len(file.loc[file['Feature'] == f])
            print(num_pair)
            
        else: 
            sum_value = file.loc[file['Feature'] == f, 'Coefficient'].sum()
            print(sum_value)
            std = round(file.loc[file['Feature'] == f, 'Coefficient'].std(),3)
            num_pair = len(file.loc[file['Feature'] == f])
   
        final_value.append(round(sum_value/num_pair,3))
        final_feat.append(f)
        final_std.append(std)
        
        # % of appearance
        presence_percent = "{:.0%}".format(round(num_pair/(len(all_lang) *3),2))
        final_appear.append(presence_percent)
        #print('final appear',final_appear)
        
    
    return final_feat, final_value, final_std, final_appear




# for coef table
final_feat, final_value, final_std, final_appear = average_coef_rank(file, rank=False)
df = pd.DataFrame({'Feature name':final_feat,'Ave. coefficient':final_value,'Std':final_std, 'Presence':final_appear})
df = df.sort_values(by=['Ave. coefficient'])

#for positional ranking  
# final_feat, final_value, final_std, final_appear = average_coef_rank(file, rank=True)
# df = pd.DataFrame({'Feature name':final_feat,'Ave. rank':final_value,'Std':final_std, 'Presence':final_appear})
# df = df.sort_values(by=['Ave. rank'])

df.to_csv("/data/s3619362/coefficient_table/coef_table_mtht.csv")





