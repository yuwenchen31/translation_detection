#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:49:57 2020

@author: chenfish
"""

# This file uses corpy udpipe to get pos tags and create training data for language models of pos.
# for DE, EN, RU, LT 


from corpy.udpipe import Model
from conllu import parse

data = []

# turn sentences into a list
with open ('/Users/chenfish/Desktop/Thesis/Project/data/news_crawl/2m_de', 'rb') as f:
        for line in f.readlines():
            data.append(str(line, 'utf-8'))
      
            
model = 'german-hdt-ud-2.5-191206.udpipe'

m = Model('../udpipe_model/'+model)
print(model, "loaded successfully!")

    
all_pos = []


for line in data:
    #print(line)
    sent_pos = []


    sents = list(m.process(line, out_format="conllu"))


    conllu = "".join(sents)
    parse_con = parse(conllu)
    
    
    # iterate over each word and append the POS into a list, 
    
    for i in parse_con[0]:
        #print(i)
        sent_pos.append(i['upostag'])

        
    # append sent pos to the the doc 
    all_pos.append(sent_pos)



# write pos list into a file 
with open('de_pos','wb') as f:
    for line in all_pos: 
        f.write(line)
    f.close()