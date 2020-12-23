#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:20:48 2020

@author: chenfish
"""

import sys
sys.path.append("../")

import pandas as pd
import os
from util.utils import corpy_udpipe

#this file is used to process the corpus (pos, universal dependency relations, head, tokenize) by UDPipe 
# and append 4 columns (POS, UD, HEAD, TOK) to the dataframe 



data_path = '/Users/yuwen/Desktop/Thesis/Project/data/ht_pe/all_no_split/mtht/'
print(data_path)
print('We are working on pos tagging & ud.')

#data_path = '/Users/chenfish/Desktop/Thesis/Project/data/mt_ht/subset_deepl/dev/'

for i in os.listdir(data_path): 
    
    try: 
    
        if not i.startswith('.'):
    
            train = pd.read_pickle(data_path + i)
            print('Now we are working on', i)
            
        else:
            print('Skip the dot file.')
            continue
        
    except IsADirectoryError:
        continue
        

    content = train['TRAIN']
    
    if i[-2:] == 'ru':
        print('Load RU udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='russian-syntagrus-ud-2.5-191206.udpipe')
        
        
    elif i[-2:] == 'de': 
        print('Load DE udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='german-hdt-ud-2.5-191206.udpipe')
        
        
    elif i[-2:] == 'lt': 
        print('Load LT udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='lithuanian-alksnis-ud-2.5-191206.udpipe')
        
        
    elif i[-2:] == 'fr': 
        print('Load FR udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='french-gsd-ud-2.5-191206.udpipe')
        
    elif i[-2:] == 'nl':
        print('Load NL udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='dutch-alpino-ud-2.5-191206.udpipe')
    
    elif i[-2:] == 'pt':
        print('Load PT udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='portuguese-gsd-ud-2.5-191206.udpipe')
        
    elif i[-2:] == 'fi':
        print('Load FI udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False, model='finnish-ftb-ud-2.5-191206.udpipe')
    
    else: 
        print('Load EN udpipe model.')
        pos, head, dep, tok = corpy_udpipe(content,sent_level=False)
     
    
    
    #add three column 
    print('Writing in pos tags, UD, and head...')
    train = train.assign(POS=pos, UD=dep, HEAD=head, TOK=tok)
    
    
    train.to_pickle(data_path + i)
    print('Saved Successfully!')
    
        
# with open ('./three paragraphs','r') as f:
#     data = f.read()
#     sents = list(m.process(data,in_format="horizontal", out_format="conllu"))
#     print("".join(sents),end="")
#     #print(data)
    
