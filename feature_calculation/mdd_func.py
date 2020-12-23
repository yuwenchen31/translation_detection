#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:47:16 2020

@author: chenfish
"""

# MDD 

import spacy_udpipe
from nltk.tokenize import word_tokenize

#mockup, list of sents
training_data = ['Churkin said, that the UN Security, Council meeting on Crimea was useful.']
                 


#load the UD model of English
ud_model = spacy_udpipe.load("en")

sent_pos = []
sent_ud = []
sent_head = []   


# get pos and ud tag
for line in training_data:

    #print(line)
    temp_pos = []
    temp_ud = []
    temp_head = []
    #print(line)
    tag_sent = ud_model(line)
    for i, token in enumerate(tag_sent):
        temp_pos.append(token.pos_)
        temp_ud.append(token.dep_)
        
        
        # if token.head == token:
        #     head = 0 
        #     temp_head.append(head)
        #     print(token, token.head, head)
        # else:

        head = token.head.i - tag_sent[0].i + 1 
        temp_head.append(head)
        print(token, token.head, head)
            
    sent_pos.append(temp_pos)
    sent_ud.append(temp_ud)
    sent_head.append(temp_head)

    #print(sent_pos)
    #print(sent_head)
    

    
    
    
def cal_mdd(tok, head, pos):
    
    #s_tok = word_tokenize(sent)
    
    #old_id = [i for i in range(1,len(sent)+1)]
    
    # remove punct from head & sent list 
    punct_indices = [i for i, x in enumerate(pos) if x == "PUNCT"]
    head_wo_punct = [i for j, i in enumerate(head) if j not in punct_indices]
    sent_wo_punct = [i for j, i in enumerate(tok) if j not in punct_indices]       
    
    
    new_head = []
    
    for i, j in enumerate(head_wo_punct): # j is the word index + 1, old-id of the head. i is the word index
        
    
        print('j: ', j)
        if j == 0:
            head_word = tok[i]
            print('head word: ', head_word)
            
        else: 
            head_word = tok[j-1] # find the original word, using old-id to index the original sent
            print('head_word: ', head_word)
        
        try: 
            
            new_word_id = sent_wo_punct.index(head_word) + 1 #new_word_id is index + 1 as the old head
            new_head.append(new_word_id)
            
        
        except ValueError: 
            print('This word: ', head_word, 'is the head of another word, but punct should not be incuded in calculating MDD. This wrong relation will be discarded.')
            pass
        print('new_word_id: ', new_word_id)
        
        
    new_id = [i for i in range(1,len(new_head)+1)]
    
    
    try: 
    
        mdd = sum([abs(i-j) for i,j in zip(new_id,new_head)]) / (len(sent_wo_punct) - 1)

    except ZeroDivisionError: 
    
        print('ZeroDivisionError! MDD will be 0.')
        mdd = 0
        
    
        
    return mdd
        
        
        
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    