#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:38:19 2020

@author: chenfish
"""

# edit distance for mt/pe files
# calculate edit distance for each sentences, average them for each chunk
# append a column to the df 
# plot the result for each lang-pair


import editdistance
import pandas as pd 
import os
import matplotlib.pyplot as plt


data_path = "/Users/chenfish/Desktop/Thesis/Project/Data/mt_pe/train/"

# for d in os.listdir(data_path): 
    
    
    
#     if not d.startswith('.'):
file = pd.read_pickle(data_path + 'deen')
print('Now we are working on', 'deen')

mt = file['TRAIN'].loc[file['LABEL'] == 'MT']
pe = file['TRAIN'].loc[file['LABEL'] == 'PE']                    

edit_doc = []

for doc_m, doc_p in zip(mt,pe):
    
    edit_sum = 0
    
    for line_m, line_p in zip(doc_m, doc_p):
        #print(editdistance.eval(line_m, line_p))
        
        edit_sum += editdistance.eval(line_m, line_p)
    
    #print('sum is:', edit_sum)
    #print('len of the doc:', len(doc_m))
    #print('average edit distance is:', edit_sum/len(doc_m))
    
    edit_doc.append(edit_sum/len(doc_m))
            
    # else: 
    #     print('Skip the file.', d)
    #     continue
        

fig = plt.figure()
ax = fig.add_subplot()
plt.ylabel("edit distance")
plt.title('DE-EN')
ax.boxplot(edit_doc, notch = True) #showfliers=False)


    