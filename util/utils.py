#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 19:05:51 2020

@author: chenfish
"""

import sys
import os
from bs4 import BeautifulSoup
import spacy_udpipe
from conllu import parse
from corpy.udpipe import Model
from nltk import tokenize


def read_corpus(filename):
    

    root = os.path.dirname(os.path.abspath('__file__'))
    
    
    raw_data = []
    

    if 'ref' in filename: 
        file_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/ht/' + filename
        
    elif 'src'in filename:
        file_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/src/' + filename
        
    else:
        file_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/mt/' + str(filename)[-4:] + '/' + filename
        


        
    with open(file_dir, 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            raw_data.append(str(line, 'utf-8'))
        
    return raw_data




def get_orig(mt_file,ht_file,src_file,language, document_level=False):
    
    #for sgm file, to get the non-translationese part 
     
    #root = os.path.dirname(os.path.abspath('__file__'))
    
    ht_sents = []
    mt_sents = []
    src_sents = []
    docid = []
    
    
    ht_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/ht/' + ht_file
    src_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/src/' + src_file
    mt_dir = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/mt/' + str(mt_file)[-8:-4] + '/' + mt_file
    
    mt_soup = BeautifulSoup(open(mt_dir),'html.parser')    
    ht_soup = BeautifulSoup(open(ht_dir),'html.parser')
    src_soup = BeautifulSoup(open(src_dir),'html.parser')
    
    orig_src = src_soup.find_all(origlang=language)
    
    
    if document_level:
        
        for doc in orig_src:
            temp_doc = []
            for line in doc.find_all('seg'): 
                temp_doc.append(line.text)
            src_sents.append(temp_doc)
            docid.append(doc.get('docid'))

        for i in docid:
            mt_doc = mt_soup.find(docid=i)
            ht_doc = ht_soup.find(docid=i)
            
            temp_mt =[]
            temp_ht = []
        
            for line in mt_doc.find_all('seg'):
                temp_mt.append(line.text)
            
            for line in ht_doc.find_all('seg'):
                temp_ht.append(line.text)
            
            mt_sents.append(temp_mt)
            ht_sents.append(temp_ht)          
        
  
    else: 
    
        # get orig in src
        for doc in orig_src:
            for line in doc.find_all('seg'): 
                src_sents.append(line.text)
            docid.append(doc.get('docid')) #get the docid list
    
        
        # use docid to find the document in mt and src 
        for i in docid:
            mt_doc = mt_soup.find(docid=i)
            ht_doc = ht_soup.find(docid=i)
            
            mt_a = mt_doc.find_all('seg')
            ht_a = ht_doc.find_all('seg')
            
            for line in mt_a:
                mt_sents.append(line.text)
            
            for line in ht_a:
                ht_sents.append(line.text) 
            
    
    return mt_sents, ht_sents, src_sents


def subset_sentences(sentence_list, num=10):
    
    #input is lists from get_orig (sentence level), each item has a sentence
    #they will be converted into lists, each item has [num] sentence

    
    
    subset = [sentence_list[i:i+num] for i in range(0,len(sentence_list),num)]
    
    
    return subset




def generate_file_name(year, lang_pair, sgm=True): 
    
    #sgm is needed before 2018 to get the non-translationese part 
    
    mt_path = '/Users/yuwen/Desktop/Thesis/Project/data/mt_ht/raw_data/mt/' +  str(lang_pair)
    
    if sgm: 
        mt_file = [i for i in os.listdir(mt_path) if str(year) in i and 'sgm' in i][0]
        ht_file = 'newstest' + year + '-' + lang_pair + '-ref.' + lang_pair[-2:] + '.sgm'
    
        src_file = 'newstest' + year + '-' + lang_pair + '-src.' + lang_pair[:2] + '.sgm'
        
        
    else: 
        mt_file = [i for i in os.listdir(mt_path) if str(year) in i][0]
        ht_file = 'newstest' + year + '-' + lang_pair + '-ref.' + lang_pair[-2:]
    
        src_file = 'newstest' + year + '-' + lang_pair + '-src.' + lang_pair[:2]



    return mt_file, ht_file, src_file



def corpy_udpipe(text,sent_level=True, model='english-lines-ud-2.5-191206.udpipe'):

    m = Model('../udpipe_model/'+model)
    print(model, "loaded successfully!")

    if sent_level: 
        
        all_pos = []
        all_head = []
        all_dep = []
        all_tok = []
        
        for line in text:
            #print(line)
            sent_pos = []
            sent_head = []
            sent_dep = []
            sent_tok = []

            sents = list(m.process(line, out_format="conllu"))
        

            conllu = "".join(sents)
            parse_con = parse(conllu)
            
            # iterate over each word and append the POS/HEAD/UD into a list, 
            
            #print(parse_con[0])
            
            for i in range(len(parse_con)):
                for word in parse_con[i]:
                    #print(i)
                    sent_pos.append(word['upostag'])
                    sent_head.append(word['head'])
                    sent_dep.append(word['deprel'])
                    sent_tok.append(word['form'])
                
            # append sent pos to the the doc 
            all_pos.append(sent_pos)
            all_head.append(sent_head)
            all_dep.append(sent_dep)
            all_tok.append(sent_tok)
    
    # for doc-level 
    else:
        
        all_pos = []
        all_head = []
        all_dep = []
        all_tok = []
        

        for doc in text:
            
            pos_per_doc = []
            head_per_doc = []
            dep_per_doc = []
            tok_per_doc = []
            
            for line in doc:
                #print(line)
                sent_pos = []
                sent_head = []
                sent_dep = []
                sent_tok = []
                
                sents = list(m.process(line, out_format="conllu"))
                conllu = "".join(sents)
                parse_con = parse(conllu)
                
                # iterate over each word and append the POS/HEAD/UD into a list, 
                
                #print(parse_con[0])
                
                for i in range(len(parse_con)):
                    for word in parse_con[i]:
                        #print(i)
                        sent_pos.append(word['upostag'])
                        sent_head.append(word['head'])
                        sent_dep.append(word['deprel'])
                        sent_tok.append(word['form'])
        
                # append sent pos to the the doc 
                
                pos_per_doc.append(sent_pos)
                head_per_doc.append(sent_head)
                dep_per_doc.append(sent_dep)
                tok_per_doc.append(sent_tok)
                
            all_pos.append(pos_per_doc)
            all_head.append(head_per_doc)
            all_dep.append(dep_per_doc)
            all_tok.append(tok_per_doc)
        
    
    return all_pos, all_head, all_dep, all_tok
    


def make_training_and_labels(item1,item2, label = ['MT', 'HT']):
    
    training = item1 + item2
    # for line in training:
    #     line = line.lower()
    label_list = [] 
    
        
    label_list = [label[0]] * len(item1) + [label[1]] * len(item2)
    
    
    return training, label_list



def get_pos_ud_head(training_data, lang='en', document=None):
    
    try:
        ud_model = spacy_udpipe.load(lang)
        print(lang,' model is used.')
        
    except Exception:
        
        spacy_udpipe.download(lang)
        print('downloaded model: ', lang)
        
        ud_model = spacy_udpipe.load(lang)
    

    sent_pos = []
    sent_ud = []
    sent_head = []  
    sent_tok = []
    
    
    # get pos and ud tag
    for line in training_data:
    
        #print(line)
        temp_pos = []
        temp_ud = []
        temp_head = []
        temp_tok = []
        #print(line)
        tag_sent = ud_model(line)
        for i, token in enumerate(tag_sent):
            temp_pos.append(token.pos_)
            temp_ud.append(token.dep_)
            temp_tok.append(token.text)

            # if token.head == token:
            #     head = 0 
            #     temp_head.append(head)
            #     print(token, token.head, head)
            # else:
    
            head = token.head.i - tag_sent[0].i + 1 
            temp_head.append(head)
            #print(token, token.head, head)
                
        sent_pos.append(temp_pos)
        sent_ud.append(temp_ud)
        sent_head.append(temp_head)
        sent_tok.append(temp_tok)
    
    return sent_pos, sent_ud, sent_head, sent_tok
    
    
    
    
    
    