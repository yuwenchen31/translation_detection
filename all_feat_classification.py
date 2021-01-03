#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:11:38 2020

@author: chenfish
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from nltk.tokenize import word_tokenize
from builtins import sum
import string
import math
import nltk
from nltk.collocations import *
import itertools
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import time
import sys
sys.path.append("../")


#__________________________________________________________________________________________________



# SIMPLIFICATION ===========================================

class LexicalDensity(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def lexical_density(self,sent_pos):
    
        #flatten the list of list 
        flatten_pos = [item for sublist in sent_pos for item in sublist]

        #calculate the lexical density 
        pos_count = Counter(flatten_pos)
        LD = sum(pos_count[x] for x in ['ADV','ADJ','NOUN','VERB'])/len(flatten_pos)
        
        return LD
    
    
    
    def transform(self,X):

        
        X_pos = pd.DataFrame(X['POS'].map(lambda x: self.lexical_density(x)))
        #print(X_pos)
        
        return X_pos
    
    
    def fit(self, X, y=None):
        return self
    
    def get_feature_names(self):
        return ['LexDen']
    

  
    
class TypeTokenRatio(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def TTR(self,sent_tok):
        
        #flatten the list of list 
        flatten_tok = [item for sublist in sent_tok for item in sublist]
        
        ttr = len(set(flatten_tok)) / len(flatten_tok)
    
        return ttr
    
    
    def transform(self,X):
        X_ttr = pd.DataFrame(X['TRAIN'].map(lambda x:self.TTR(x)))
        return X_ttr
    
    
    def fit(self, X, y=None):
        return self
    
    def get_feature_names(self):
        return ['TTR']

    

class YulesI(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def get_yules(self,sent_tok):
        
        tokens = [item for sublist in sent_tok for item in sublist]
        #print(tokens)

        token_counter = Counter(tok.lower() for tok in tokens)
        
        m1 = sum(list(token_counter.values()))

        m2 = sum([freq ** 2 for freq in token_counter.values()])

        try:
            return (m1*m1) / (m2-m1) # this is i
            
        except:
            return 0  
            
        #k = 1/i * 10000
    
        
    
    def transform(self,X):
        X_yule = pd.DataFrame(X['TOK'].map(lambda x:self.get_yules(x)))
        return X_yule
    
    
    def fit(self, X, y=None):
        return self
    
    def get_feature_names(self):
        return ['yulesI']


class MTLD(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        self.remove_punctuation = str.maketrans('', '', string.punctuation)

    # MTLD internal implementation
    def mtld_calc(self, word_array, ttr_threshold):
        current_ttr = 1.0
        token_count = 0
        type_count = 0
        types = set()
        factors = 0.0
        
        for token in word_array:
    # AT do not lowercase
    #        token = token.translate(remove_punctuation).lower() # trim punctuation, make lowercase
    
    # AT keep punctuation (for fair comparison with other methods that keep it)
            token = token.translate(self.remove_punctuation)
    #        token = token.lower()
    
            token_count += 1
            if token not in types:
                type_count +=1
                types.add(token)
            current_ttr = type_count / token_count
            if current_ttr <= ttr_threshold:
                factors += 1
                token_count = 0
                type_count = 0
                types = set()
                current_ttr = 1.0
        
        excess = 1.0 - current_ttr
        excess_val = 1.0 - ttr_threshold
        factors += excess / excess_val
        if factors != 0:
            return len(word_array) / factors
        return -1
    
    # MTLD implementation
    def mtld(self, word_array, ttr_threshold=0.72):
        #print(word_array)
        
        tokens = [item for sublist in word_array for item in sublist]
        
        if isinstance(tokens, str):
            raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
        #if len(word_array) < 50:
            #raise ValueError("Input word list should be at least 50 in length")
        return (self.mtld_calc(tokens, ttr_threshold) + self.mtld_calc(tokens[::-1], ttr_threshold)) / 2

    
        
    
    def transform(self,X):
        X_mtld = pd.DataFrame(X['TOK'].map(lambda x:self.mtld(x)))
        #print(X_mtld)
        
        return X_mtld
    

    
    
    def fit(self, X, y=None):
        return self

class AveWordLength(BaseEstimator, TransformerMixin):
    
    
    
    def __init__(self):
        pass
    
    
    def mean_word_length(self,tok_sentence): #character-level: translated might use shorter words (simplification)
    
        flat_tok = [item for sublist in tok_sentence for item in sublist]
        #print(flat_tok)
        ave_word_len = sum([len(w) for w in flat_tok]) / len(flat_tok) #sum all words length/# of tokens
    
        return ave_word_len
    
    def transform(self,X):
        
        X_length = pd.DataFrame(X['TOK'].map(lambda x: self.mean_word_length(x)))
        #print(X_length[:10])
        #print(X_length)
        return X_length
    
    def fit(self, X, y=None):
    
        return self    
    
    def get_feature_names(self):
        return ['average word length']
    
class LengthRatio(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        pass
        
        
    
    def length_ratio(self,tar_sent,src_sent):
        
        #flatten the list 
        #flatten_tar = [item for sublist in tar_sent for item in sublist]
        #flatten_src = [item for sublist in src_sent for item in sublist]
        #print(flatten_tar)
        tar_sent = ' '.join(tar_sent)
        src_sent = ' '.join(src_sent)
      
        t_tok = word_tokenize(tar_sent)
        s_tok = word_tokenize(src_sent)
        #print(t_tok)
        
        #calculate length in characters 
        len_tar = sum([len(w) for w in t_tok])
        len_src = sum([len(w) for w in s_tok])
        
        #len_ratio = abs(len(src_sent)-len(t_tok))/len(src_sent)
            
        len_ratio = abs(len_src - len_tar)/len_src
    
        return len_ratio
    
    def transform(self,X):
        
        # LR_all = []
        # for tar_sent,src_sent in zip(X['Training'],X['SRC']):
        #     LR_per_sent = abs(len(src_sent)-len(tar_sent))/len(src_sent)
        #     LR_all.append(LR_per_sent)
        #X_LR = pd.DataFrame(data={"Length Ratio":LR_all})
        X_LR = pd.DataFrame(X.apply(lambda x: self.length_ratio(x['TRAIN'],x['SRC']), axis=1))
        
        #print(X_LR[:5])
        
        return X_LR
    
    def fit(self, X, y=None):
        return self


class MDD(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
        
    
    def cal_mdd(self, tok, head, pos):
        
        
        #flat_tok = [item for sublist in tok for item in sublist]
        #flat_head = [item for sublist in head for item in sublist]
        #flat_pos = [item for sublist in pos for item in sublist]
        
        #print(tok)
        
        mdd = 0
        
        for t,h,p in zip(tok,head,pos):
            
            #print(t,h,p)
            
            # remove punct from head & sent list 
            punct_indices = [i for i, x in enumerate(p) if x == "PUNCT"]
            head_wo_punct = [i for j, i in enumerate(h) if j not in punct_indices]
            sent_wo_punct = [i for j, i in enumerate(t) if j not in punct_indices]       
            
            #print(punct_indices)
            #print(head_wo_punct)
            #print(sent_wo_punct)
            
            
            new_head = []
            
            for i, j in enumerate(head_wo_punct): # j is the word index + 1, old-id of the head. i is the word index
                
                
                #print('j: ', j)
                if j == 0:
                    head_word = t[i]
                    #print('head word: ', head_word)
                    
                else: 
                    
                    try: 
                        head_word = t[j-1] # find the original word, using old-id to index the original sent
                        #print('head_word: ', head_word)

                
                        new_word_id = sent_wo_punct.index(head_word) + 1 #new_word_id is index + 1 as the old head
                        new_head.append(new_word_id)


                    except: 
                        #print('This word: ', head_word, 'is the head of another word, but punct should not be incuded in calculating MDD. This wrong relation will be discarded.')
                        #print(tok)
                        #print(head)
                        #print(pos)
                        pass
                #print('new_word_id: ', new_word_id)
                
                
            new_id = [i for i in range(1,len(new_head)+1)]
            
            
            try: 
            
                mdd = sum([abs(i-j) for i,j in zip(new_id,new_head)]) / (len(sent_wo_punct) - 1)
        
            except ZeroDivisionError: 
            
                #print('ZeroDivisionError! MDD will be 0.')
                mdd = 0
            
            
            mdd += mdd 
        
        final_mdd = mdd/len(tok)
        
        
        return final_mdd
        
    def transform(self,X):
        
        X_MDD = pd.DataFrame(X.apply(lambda x: self.cal_mdd(x['TOK'],x['HEAD'],x['POS']), axis=1))
        
        #print(X_MDD[:5])
        
        return X_MDD
    
    def fit(self, X, y=None):
        return self

class AveSentenceLength(BaseEstimator, TransformerMixin):
    
    
    
    def __init__(self):
        pass
    
    
    def mean_sent_length(self,tok_sentence): #translation might use shorter sentences
    
        sent_len = 0
        
        for sent in tok_sentence:
            sent_len += len(sent)
        
        ave_sent_len = sent_len / len(tok_sentence) #sum all sent length/# of sents
        #print(ave_sent_len)
        return ave_sent_len
    
    def transform(self,X):
        
        X_length = pd.DataFrame(X['TOK'].map(lambda x: self.mean_sent_length(x)))
        #print(X_length[:10])
        #print(X_length)
        return X_length
    
    def fit(self, X, y=None):
    
        return self   


#syllable ratio: only for EN so far 



# N most frequent words
# Most frequent words The normalized fre- quencies of the N most frequent words in the corpus. 
#We define three features, with three different thresholds: N 1⁄4 5,10,50. 
#Punctuation marks are excluded.




# EXPLICITATION ===========================================

#explicit naming 
# # of pronouns/# of proper nouns
# MT will use more proper nouns: pronoun ratio will be smaller
# HT use more pronouns, pronoun ratio will be bigger


class ExplicitNaming(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def pronoun_ratio(self,sent_pos):
        
        flat_pos = [item for sublist in sent_pos for item in sublist]
    
        pos_count = Counter(flat_pos)
        try: 
            pronoun_ratio = pos_count['PRON'] / pos_count['PROPN']
            
        except ZeroDivisionError:
            pronoun_ratio = pos_count['PRON']
        
        return pronoun_ratio
    
    def transform(self,X):

        
        X_prop = pd.DataFrame(X['POS'].map(lambda x: self.pronoun_ratio(x)))
        #print(X_ud)
        
        return X_prop
    
    def fit(self, X, y=None):
        return self

class SingleNaming(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def single_proper_noun_counts(self,sent_pos):
        
        flat_pos = [item for sublist in sent_pos for item in sublist]
            
        count = 0
        for i,j in enumerate(flat_pos):

            try: 
                if j == 'PROPN' and flat_pos[i-1] != 'PROPN' and flat_pos[i+1] != 'PROPN':
                    count += 1
            
            # if indexerror:  it means 'PROPN' is at the last position in the flat_pos, 
            # so count will +1
            except IndexError:
                count += 1
                
        
        return count
    
    def transform(self,X):

        
        X_sin_naming = pd.DataFrame(X['POS'].map(lambda x: self.single_proper_noun_counts(x)))
        #print(X_ud)
        
        return X_sin_naming
    
    def fit(self, X, y=None):
        return self


#??? 
class MeanMultipleNaming(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def mean_multiple_propn(self,doc_pos):
        
        #flat_pos = [item for sublist in sent_pos for item in sublist]
                
        len_propn = []
        
        for sent in doc_pos: 
            for k,g in itertools.groupby(sent):
                p = list(g)
                
                if 'PROPN' in p and len(p) > 1: 
                    #print(p)
                    #print('found one multiple naming.')
                    len_propn.append(len(p))
                    
             
        try:
            return sum(len_propn)/len(len_propn)
            
        except:
            return 0
                    
    
    def transform(self,X):

        
        X_sin_naming = pd.DataFrame(X['POS'].map(lambda x: self.mean_multiple_propn(x)))
        #print(X_ud)
        
        return X_sin_naming
    
    def fit(self, X, y=None):
        return self



class AveFunctionWord(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def ave_function_word(self,sent_ud):
        
        flat_ud = [item for sublist in sent_ud for item in sublist]
            
        
        ud_count = Counter(flat_ud)
        ave_func_word = sum(ud_count[x] for x in ['aux','cop','mark','det','clf','case'])/len(flat_ud)
        
        return ave_func_word
    
    def transform(self,X):

        
        X_ud = pd.DataFrame(X['UD'].map(lambda x: self.ave_function_word(x)))
        #print(X_ud)
        
        return X_ud
    
    def fit(self, X, y=None):
        return self


# Normalization ===========================================
    
class RepetitionRatio(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def repetition(self,sent_pos,sent_tok):
        
        #flatten the list of list
        flatten_pos = [item for sublist in sent_pos for item in sublist]
        flatten_tok = [item for sublist in sent_tok for item in sublist]
        
        
        # # of content words that occur > 1/# tokens
        # content words: words that are tagged as nouns, verbs, adjectives, and adverbs
        
        #find words are tagged as nouns, verbs, adjectives, or adverbs
        content_tok = []
        for p,t in zip(flatten_pos, flatten_tok): 
            if p in ['ADV','ADJ','NOUN','VERB']: 
                content_tok.append(t)
        
        #dic{token: counts of occurring times}
        all_content_token_count = Counter(content_tok)
        
        reoccur_content_count = 0
        
        for c in all_content_token_count.values(): 
            if c > 1:
                reoccur_content_count += c
        
        
        repetition_ratio = reoccur_content_count / len(flatten_tok)
    
        return repetition_ratio
    
    def transform(self,X):
        X_Re = pd.DataFrame(X.apply(lambda x: self.repetition(x['POS'],x['TOK']), axis=1))
        return X_Re
    
    def fit(self, X, y=None):
        return self





# Interference ===========================================


class ColumnExtractor(BaseEstimator,TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #print(f'{X[self.column].shape} X[pos] shape')
        #print(f'{X[self.column]} X[pos] contents')
        #print(f'{type(X[self.column])} X[pos] type')
        return X[self.column] #.values.reshape(-1,1)
    
    
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, data_frame):
        # new_df = data_frame.values.reshape(-1,1)
        # print(new_df.shape)
        return data_frame.values.reshape(-1,1)
    
    
class String_Converter(BaseEstimator, TransformerMixin):
    
    #for tfidf-text: TRAIN is a list of string(sentence)
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print(type(X)) #pandas.series
        
        X_str = pd.DataFrame(X.map(lambda x:' '.join(x)))
        
        #print(X_str)
        
        #print(type(X_str)) #DataDrame
        #print(f'{X_str.values.reshape(-1)} X_str shape')
        new = X_str.values.reshape(-1)
        #print(new.shape)
        #print(type(new))
        #print(new)
        return new
    


class List_Converter(BaseEstimator, TransformerMixin):
    
    #for tfidf-pos: POS is list of list(POS) 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print(type(X)) #pandas.series
        #print(X)
        #print(X.shape, 'X_shape')
        
        
        X_str = pd.DataFrame(X.map(lambda x: ' '.join(l for doc in x for l in doc)))
        
        #print(type(X_str)) #DataDrame
        #print(f'{X_str.values.reshape(-1)} X_str shape')
        new = X_str.values.reshape(-1)
        #print(new.shape)
        #print(type(new))
        #print(new)
        return new


#character ngrams
tfidf_text = Pipeline([
                        ('tfidf',ColumnExtractor(column='TRAIN')),
                        ('Stringconvert', String_Converter()),
                        #('tfidf', TfidfVectorizer(stop_words='english', analyzer = 'char', ngram_range=(1,3)))
                        ('vec', TfidfVectorizer(analyzer = 'char', ngram_range=(2,3), max_features=1000))
            ])



#pos ngrams
tfidf_pos = Pipeline([
                        ('pos ngram',ColumnExtractor(column='POS')),
           
                        ('string convert', List_Converter()),
                        ('vec', TfidfVectorizer(analyzer = 'word', ngram_range=(2,2)))
            ])



#perplexity of pos sequence



# Others ===========================================

#Kendall's tau distance
#Pronouns
class PronounRatio(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def pronoun_ratio(self,sent_pos):
        
        flat_pos = [item for sublist in sent_pos for item in sublist]
    
        pos_count = Counter(flat_pos)

        pronoun_ratio = pos_count['PRON'] / len(flat_pos)
            

        
        return pronoun_ratio
    
    def transform(self,X):

        
        X_prop = pd.DataFrame(X['POS'].map(lambda x: self.pronoun_ratio(x)))
        #print(X_ud)
        
        return X_prop
    
    def fit(self, X, y=None):
        return self



#Ratio of passive forms to all verbs (only in english translation)

#AUX + VERB

class PassiveVerbRatio(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    
    def pass_verb_ratio(self,doc_pos):
        
        #ori_flat_pos = [item for sublist in doc_pos for item in sublist ]
        #ori_pos_count = Counter(ori_flat_pos)
        
        #print('before VERB count',ori_pos_count['VERB'])
        
        pass_count = 0
        for sent in doc_pos:
            #print(sent)
            for ind,word in reversed(list(enumerate(sent))):
                
                #looping from backwards, and delete items from backwards
                #check if its VERB
                
                if word == 'VERB':
                    if sent[ind-1] == 'AUX':
                        pass_count += 1
                        #print(sent)
                        #print(ind)
                        #print(sent[ind])
                       
                    elif sent[ind-1] == 'ADV' and sent[ind-2] == 'AUX':
                        pass_count += 1
                        #print(sent)
                        #print(ind)
                        #print(sent[ind])
                       

                else:
                    continue
                    
                    
        flat_pos = [item for sublist in doc_pos for item in sublist ]
        pos_count = Counter(flat_pos)
        #print('pass_count',pass_count)
        #print('after VERB count',pos_count['VERB'])
        
        try:
            return pass_count / pos_count['VERB']
        
        except ZeroDivisionError:
            return 1
                    
    
    def transform(self,X):

        
        X_prop = pd.DataFrame(X['POS'].map(lambda x: self.pass_verb_ratio(x)))
        #print(X_ud)
        
        return X_prop
    
    def fit(self, X, y=None):
        return self





# Standard Scaler ========================================

#Q: what does standardscaler do? 

lex_den = Pipeline([ ('Lexical Density', LexicalDensity()),
                         ('standard', StandardScaler())
                      ])

word_length = Pipeline([ ('Mean Word Length', AveWordLength()),
                         ('standard', StandardScaler())
                      ])


func_word = Pipeline([ ('Average Function Words', AveFunctionWord()),
                         ('standard', StandardScaler())
                      ])


ttr =  Pipeline([ ('Type Token Ratio', TypeTokenRatio()),
                  ('standard', StandardScaler())
                      ])


yulesi = Pipeline([ ('yules I', YulesI()),
                    ('standard', StandardScaler())
                      ])


length_ratio = Pipeline([ ('Length Ratio', LengthRatio()),
                         ('standard', StandardScaler())
                      ])

MDD = Pipeline([ ('Mean dependency distance', MDD()),
                 ('standard', StandardScaler())
                      ])

Repetition = Pipeline([ ('Repetition', RepetitionRatio()),
                       ('standard', StandardScaler())
                      ])

explicit_naming = Pipeline([ ('Explicit Naming', ExplicitNaming()),
                             ('standard', StandardScaler())
                      ])

single_naming = Pipeline([ ('Single Naming', SingleNaming()),
                             ('standard', StandardScaler())
                      ])

perple_pipe =  Pipeline([
                        ('perplexity',ColumnExtractor(column='PER')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

word_rank_pipe = Pipeline([
                        ('mean word rank',ColumnExtractor(column='RANK')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

pmi_pipe = Pipeline([
                        ('PMI',ColumnExtractor(column='PMI')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


syl_pipe = Pipeline([
                        ('syllable ratio',ColumnExtractor(column='SYL')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

threpmi_pipe = Pipeline([
                        ('thresold PMI',ColumnExtractor(column='ThresoldPMI')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


freq50_pipe = Pipeline([
                        ('50 freq words',ColumnExtractor(column='freq 50')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

freq10_pipe = Pipeline([
                        ('10 freq words',ColumnExtractor(column='freq 10')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

freq5_pipe = Pipeline([
                        ('10 freq words',ColumnExtractor(column='freq 5')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

perdiff_pipe = Pipeline([
                        ('perplexity difference',ColumnExtractor(column='PER diff')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

confunc_pipe = Pipeline([
                        ('context func',ColumnExtractor(column='CONTEXT FUNC')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

ptf_pipe = Pipeline([
                        ('context func',ColumnExtractor(column='PTF')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])



# Feature union connects all features ===================

if __name__ == "__main__":
    
    # clf types: mtht, htpe, mtpe
    clf_type = sys.argv[1]

    # mtht - 11 language pairs 
    if clf_type == 'mtht':
        lang_pair = ['ende', 'enfi', 'enlt', 'enru','deen','fien','guen','kken','lten','ruen','zhen']

    # htpe (zhen) - compares data in htpe and mtht 
    elif clf_type == 'htpe':
        lang_pair = ['htpe','mtht']    

    # mtpe - 5 language pairs 
    else:
        lang_pair = ['ende','enru','enfr','ennl','enpt']
    

    context_length = [2,5,10]
    
    num_features_two = []
    num_features_five = []
    num_features_ten = []
    
    accu_two = []
    accu_five = []
    accu_ten = []
    

    
    for l in lang_pair:

        
        for con_len in context_length:
            
            timer_first = time.time()
            
            train_file = 'train.' + str(n) + '.' + l
            print('train filename is:', train_file)
            
            dev_file = 'dev.' + str(n) + '.' + l
            print('dev filename is:', dev_file)
            
            test_file = 'test.' + str(n) + '.' + l
            print('test filename is:', test_file)
        

            train_file = "./data/" + clf_type +  "/" + train_file
            dev_file = "./data/" + clf_type + "/" + dev_file
            test_file = "./data/" + clf_type + "/" + test_file
        
            train = pd.read_pickle(train_file)
            dev = pd.read_pickle(dev_file)
            test = pd.read_pickle(test_file)
            
            
            
            train_column = list(train.columns)
            train_column.remove('LABEL')
            
            dev_column = list(dev.columns)
            dev_column.remove('LABEL')
            
            test_column = list(test.columns)
            test_column.remove('LABEL')
            
            
            
            X_train = train[train_column]
            y_train = train[['LABEL']].values.ravel()
            
        
            X_dev = dev[dev_column]
            y_dev = dev[['LABEL']].values.ravel()
        
            X_test = test[test_column]
            y_test = test[['LABEL']].values.ravel()
            
       
            X_test = pd.concat([X_test,X_dev],ignore_index=True)
            y_test = np.concatenate((y_test, y_dev), axis=0)
            
            
            feature_mapping = {
                
                'Character ngram': tfidf_text,
                'PoS ngram': tfidf_pos,
                
                'Type Token Ratio': ttr,
                'yules I': yulesi,   
                'MTLD': MTLD(),
                'Average Word Length': word_length,
                'Length Ratio': length_ratio,
                'Average Sentence Length': AveSentenceLength(), 
                'Lexical Density': lex_den, 
                'Mean Word Rank': word_rank_pipe,
                'MDD': MDD,
                'Perplexity': perple_pipe,
                'Syllable Ratio': syl_pipe,
                '50 most frequent words': freq50_pipe,
                '10 most frequent words': freq10_pipe,
                '5 most frequent words': freq5_pipe, 
                

                'PoS perplexity': perdiff_pipe,
                'Contextual Function Words': confunc_pipe,
                'Positional Token Frequency': ptf_pipe,
                
                'Explicit Naming': explicit_naming,
                'Single Naming': single_naming, 
                'Mean Multiple Naming': MeanMultipleNaming(),
                'Function Words Ratio': func_word,
                
                'Repetition': Repetition,
                'Average PMI': pmi_pipe,
                'Thresold PMI': threpmi_pipe,
                
                'Pronouns Ratio': PronounRatio(),
                'Passive Verb Ratio': PassiveVerbRatio(),
                

        
                }
            
            
            unwant = []
            
            if 'PER' not in train_column:
    
                unwant.append('Perplexity')
                print('We dont have perplexity. We remove it from feature list.')
                
            
            if 'PER diff' not in train_column:

                unwant.append('PoS perplexity')
                print('We dont have PoS perplexity. We remove it from feature list.')
                
            if 'SYL' not in train_column:
           
                unwant.append('Syllable Ratio')
                print('We dont have syllable ratio. We remove it from feature list.')
                
            if not set(['RANK','freq 50', 'freq 10', 'freq 5']).issubset(train_column):
          
                unwant.extend(['Mean Word Rank','50 most frequent words', '10 most frequent words', '5 most frequent words'])
                
                
                print('We dont have rank, 5, 10, 50 most frequent words. We remove it from feature list.')
                
                
            
            for item in unwant:
                del feature_mapping[item]
                print('We delete:', item)
                
                
            # make dict into a tuple in the list as tranformer list
            union_list = [(k, v) for k, v in feature_mapping.items()]
      
    
            # feature union
            feats = Pipeline([ 
                
                    ('union',FeatureUnion(union_list)
                     
                    )])
            
            
            svc = LinearSVC()
            
            
            
            rfecv = RFECV(estimator=svc, step=1, min_features_to_select=1, scoring='accuracy', n_jobs=-1,cv=StratifiedKFold(2, random_state=40, shuffle=True))
        
            
            
            pipe_rfe = Pipeline([
                
                        ('features', feats),
                             
                        ('rfe_feature_selection', rfecv),

            
                        ])
            
            
            print('We are now fitting...')
            pipe_rfe.fit(X_train,y_train)
            
            # get only translationese features 
            trans_feature = [i[0] for i in union_list][2:]
            # get character ngram features 
            charngram_features = pipe_rfe[0][0].transformer_list[0][1][2].get_feature_names()
            #pos ngram features 
            posngram_features = pipe_rfe[0][0].transformer_list[1][1][2].get_feature_names()
            
            # concatenate all features 
            all_features = charngram_features + posngram_features + trans_feature
            
                        
            support = pipe_rfe[1].support_.tolist()
            true_features = Counter(support)[True]
            
            
            all_selected_features = [x for x,y in zip(all_features,support) if y == True]
            
        
            print(all_selected_features)
            
            
            
            print('support:',len(support))
           
            
        
            y_pred = pipe_rfe.predict(X_test)
            

            
        
            
            if con_len == 2:
        
                num_features_two.append(true_features)
                accu_two.append(accuracy_score(y_test, y_pred))
                
                print('accuracy of 2 sentences:', accu_two)
                print('features of 2 sentences:', num_features_two)
                
            elif con_len == 5:     
                num_features_five.append(true_features)
                accu_five.append(accuracy_score(y_test, y_pred))
       
                print('accuracy of 5 sentences:', accu_five)
                print('features of 5 sentences:', num_features_five)
            
            else: 
                num_features_ten.append(true_features)
                accu_ten.append(accuracy_score(y_test, y_pred))

                print('accuracy of 10 sentences:', accu_ten)
                print('features of 10 sentences:', num_features_ten)
        
        
    
        
    # plot for accuracy and selected features ---------------------------------
    
    width = 0.25
    #plt.figure(figsize=(200,300))
    fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2,figsize=(12,12)) 
    #fig.subplot(211, facecolor='white')
    ax1.grid(False)
    
    # Set position of bar on X axis 
    x = np.arange(len(lang_pair)) 
    #br2 = [x + barWidth for x in br1] 
    
    
    rects1 = ax1.bar(x - width,  accu_two, width, label='2 sent', color='darkturquoise')
    rects2 = ax1.bar(x , accu_five, width, label='5 sent', color='cadetblue') 
    rects3 = ax1.bar(x + width, accu_ten, width, label='10 sent', color='powderblue')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set(ylim=[0.4,1])
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('MTHT',fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lang_pair)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax1.legend(loc="best",frameon=True, ncol=3)
    
    
    def autolabel(rects,ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(),2)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)
        
    autolabel(rects1,ax1)
    autolabel(rects2,ax1)
    autolabel(rects3,ax1)
    
    fig.tight_layout()
    
    import seaborn as sns
    
    #fig.subplot(212, facecolor='white')
    ax2.grid(False)
    
    feat1 = ax2.bar(x - width, num_features_two, width, label='2 sent', color='darkturquoise')
    feat2 = ax2.bar(x , num_features_five, width, label='5 sent', color='cadetblue')
    feat3 = ax2.bar(x + width, num_features_ten, width, label='10 sent', color='powderblue')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set(ylim=[0,1500])
    ax2.set_ylabel('Selected Feature Numbers', fontsize=14)
       
    ax2.set_xticks(x)
    ax2.set_xticklabels(lang_pair)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
        
    autolabel(feat1,ax2)
    autolabel(feat2,ax2)
    autolabel(feat3,ax2)
    plt.show()
        
        
    
        
        
