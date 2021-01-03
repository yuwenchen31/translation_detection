#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:39:44 2020

@author: yuwen
"""

# This file eliminates features (derived from main.py) whose VIF > 5 
# The classfier is initialized with the best hyperparameters from hyperparameter folder


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
import nltk
nltk.download('punkt')
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import classification_report,accuracy_score

from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import matplotlib.pyplot as plt


# extract each feature and standarize the data ---------------------------------

class ColumnExtractor(BaseEstimator,TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        return X[self.column] 
    
    
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, data_frame):

        return data_frame.values.reshape(-1,1)
    

yulesi =  Pipeline([
                        ('perplexity',ColumnExtractor(column='yulesi')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


ttr =  Pipeline([
                        ('perplexity',ColumnExtractor(column='type token ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])



mtld =  Pipeline([
                        ('perplexity',ColumnExtractor(column='mtld')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])




average_word_length =  Pipeline([
                        ('perplexity',ColumnExtractor(column='average word length')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])



length_ratio =  Pipeline([
                        ('perplexity',ColumnExtractor(column='length ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


average_sentence_length =  Pipeline([
                        ('perplexity',ColumnExtractor(column='average sentence length')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


lexical_density =  Pipeline([
                        ('perplexity',ColumnExtractor(column='lexical density')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])




mdd =  Pipeline([
                        ('perplexity',ColumnExtractor(column='mdd')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])




mean_word_rank = Pipeline([
                        ('mean word rank',ColumnExtractor(column='mean word rank')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


perplexity = Pipeline([
                        ('PMI',ColumnExtractor(column='perplexity')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


syllable_ratio = Pipeline([
                        ('PMI',ColumnExtractor(column='syllable ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


five_most_frequent_word = Pipeline([
                        ('PMI',ColumnExtractor(column='5 most frequent word')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


ten_most_frequent_word = Pipeline([
                        ('PMI',ColumnExtractor(column='10 most frequent word')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


fifty_most_frequent_word = Pipeline([
                        ('PMI',ColumnExtractor(column='50 most frequent word')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


explicit_naming = Pipeline([
                        ('PMI',ColumnExtractor(column='explicit naming')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


single_naming = Pipeline([
                        ('PMI',ColumnExtractor(column='single naming')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


mean_multiple_naming = Pipeline([
                        ('PMI',ColumnExtractor(column='mean multiple naming')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


average_function_words = Pipeline([
                        ('PMI',ColumnExtractor(column='average function words')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


repetition_ratio = Pipeline([
                        ('PMI',ColumnExtractor(column='repetition ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])



pmi = Pipeline([
                        ('PMI',ColumnExtractor(column='pmi')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


thresold_pmi = Pipeline([
                        ('PMI',ColumnExtractor(column='thresold PMI')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


pos_perplexity = Pipeline([
                        ('PMI',ColumnExtractor(column='pos perplexity')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


pronoun_ratio = Pipeline([
                        ('PMI',ColumnExtractor(column='pronoun ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

passive_verb_ratio = Pipeline([
                        ('PMI',ColumnExtractor(column='passive verb ratio')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


positional_token_frequency = Pipeline([
                        ('PMI',ColumnExtractor(column='positional token frequency')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])

contextual_function_word = Pipeline([
                        ('PMI',ColumnExtractor(column='contextual function word')),
                        ('convert', Converter()),
                        ('standard', StandardScaler())
            ])


# eliminate features whose VIF > 5
def eliminate_vif(file):
    
    
    # only do the elimination when there is more than 1 features 
    if len(file.columns) > 1: 
        
        # get the VIF of each features 
        vif_list = [variance_inflation_factor(file.values, i) for i in range(len(file.columns))] 
    
        while max(vif_list) > 5 and len(vif_list) > 2: 
            
            # delete the feature with maximum VIF from the file 
            drop_index = vif_list.index(max(vif_list))
            drop_column_name = file.columns[drop_index]
            print(drop_column_name)
            print(max(vif_list))
            file = file.drop(columns=drop_column_name,axis=1)
            
            # final feature list 
            vif_list = [variance_inflation_factor(file.values, i) for i in range(len(file.columns))] 

    # if there are less than two features left then dont use vif filter 
    else: 
        vif_list = [0] * len(file.columns)
    
    # create dataframes with features names and their individual VIF value    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = file.columns
    vif_data["VIF"] = vif_list
    left_features = file.columns.tolist()
    
    return vif_data, left_features 


if __name__ == "__main__":

    # TODO: change the location of the hyperparameter files 
    param_2 = pd.read_csv("/data/s3619362/hyperparameters/hyperparameters_2_mtht.csv")
    param_5 = pd.read_csv("/data/s3619362/hyperparameters/hyperparameters_5_mtht.csv")
    param_10 = pd.read_csv("/data/s3619362/hyperparameters/hyperparameters_10_mtht.csv")
    
    
    context_length = [2,5,10]
    
    # for plotting the figures 
    num_features_two = []
    num_features_five = []
    num_features_ten = []
    
    accu_two = []
    accu_five = []
    accu_ten = []
    
    lang_pair = []
    
    for i in range(len(param_10)):

        # get the language pairs from the hyperparameter files 
        lang = param_10.loc[i]['language_pair']
        
        lang_pair.append(lang)
        
        df_feature_ranking = pd.DataFrame()

        for con_len in context_length:
            
            
            
            timer_first = time.time()
            
            train_file = 'train.' + str(con_len) + '.' + lang
            print('train filename is:', train_file)
            
            dev_file = 'dev.' + str(con_len) + '.' + lang
            print('dev filename is:', dev_file)
            
            test_file = 'test.' + str(con_len) + '.' + lang
            print('test filename is:', test_file)


            # TODO: change the location of the data file
            train_file = "/data/s3619362/mtht/" + train_file
            dev_file = "/data/s3619362/mtht/" + dev_file
            test_file = "/data/s3619362/mtht" + test_file
            
            
        
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
            
            
            le = preprocessing.LabelEncoder()
            
            #binarize the labels 
            y_train = le.fit_transform(y_train)
            y_dev = le.fit_transform(y_dev)
            y_test = le.fit_transform(y_test)
            
        
            # concatenate the original test and dev file as final testing set
            X_test = pd.concat([X_test,X_dev],ignore_index=True)
            y_test = np.concatenate((y_test, y_dev), axis=0)

            
            
            # TODO: change the location of the feature file
            # read the features from csv file
            feature_file = pd.read_csv("/data/s3619362/coefficient_table/context_length_all_mtht.csv")
            
            # get the selected features (derived from main.py)
            predefined_features = feature_file[(feature_file['Context Length'] == con_len) & (feature_file['Language Pair'] == lang)]['Feature Name'].tolist()
            
            print("RFE selected feature numbers:",len(predefined_features))
            
            # new training data only has the selected features (in columns)
            new_X_train = X_train[predefined_features]
            
            # eliminate features whose VIF > 5
            vif_data,all_features = eliminate_vif(new_X_train)
            
            
            # maps the feature name with the functions 
            feature_mapping = {
                
                'type token ratio': ttr,
                'yulesi': yulesi,   
                'mtld': mtld,
                'average word length': average_word_length,
                'length ratio': length_ratio,
                'average sentence length': average_sentence_length, 
                'lexical density': lexical_density, 
                'mean word rank': mean_word_rank,
                'mdd': mdd,
                'perplexity': perplexity,
                'syllable ratio': syllable_ratio,
                '50 most frequent word': fifty_most_frequent_word,
                '10 most frequent word': ten_most_frequent_word,
                '5 most frequent word': five_most_frequent_word, 
                
                

                'pos perplexity': pos_perplexity,
                'contextual function word': contextual_function_word,
                'positional token frequency': positional_token_frequency,
                
                'explicit naming': explicit_naming,
                'single naming': single_naming, 
                'mean multiple naming': mean_multiple_naming,
                'average function words': average_function_words,
                'repetition ratio': repetition_ratio,
                'pmi': pmi,
                'thresold PMI': thresold_pmi,
                
                'pronouns ratio': pronoun_ratio,
                'passive verb ratio': passive_verb_ratio,
                
        
                }
            

                
            # transfromer list for Feature Union: ['feature name', feature functions]
            union_list = [(k, v) for k, v in feature_mapping.items() if k in all_features]
           
            # feature union
            feats = Pipeline([ 
                
                    ('union',FeatureUnion(union_list)
                     
                    )])
            
            # define the classifier with the found best hyperparameters 
            if con_len == 2:
                svc = LinearSVC(C=param_2.loc[i]['C'],class_weight= None if param_2.loc[i]['class_weight'] == 'None' else param_2.loc[i]['class_weight'], loss=param_2.loc[i]['loss'], max_iter=param_2.loc[i]['max_iter'] , tol=param_2.loc[i]['tol'], random_state=param_2.loc[i]['random_state'])
            
            elif con_len == 5: 
                svc = LinearSVC(C=param_5.loc[i]['C'],class_weight= None if param_2.loc[i]['class_weight'] == 'None' else param_2.loc[i]['class_weight'], loss=param_5.loc[i]['loss'], max_iter=param_5.loc[i]['max_iter'] ,tol=param_5.loc[i]['tol'], random_state=param_5.loc[i]['random_state'])
            
            else: 
                svc = LinearSVC(C=param_10.loc[i]['C'],class_weight= None if param_2.loc[i]['class_weight'] == 'None' else param_2.loc[i]['class_weight'], loss=param_10.loc[i]['loss'], max_iter=param_10.loc[i]['max_iter'] , tol=param_10.loc[i]['tol'], random_state=param_10.loc[i]['random_state'])
                
            # initialize the rfecv classificer     
            rfecv = RFECV(estimator=svc, step=1, min_features_to_select=1, scoring='accuracy', cv=StratifiedKFold(10, random_state=42, shuffle=True))
            
            # pass all features to the classifier using pipeline
            pipe = Pipeline([
                
                        ('features', feats),
                        
                        ('clf',svc)
            
                        ])
            
            
            
            print('We are now training...')
            pipe.fit(X_train,y_train)

            # predict on the unseen data
            y_pred = pipe.predict(X_test)
            
            print('Results of', lang)
            print(classification_report(y_test, y_pred))
            
            
            
            # get feature names -----------------------------------------------
            feature_num = len(pipe[0][0].transformer_list)
            feature_names = []
            
            
            for num in range(feature_num):
                name = pipe[0][0].transformer_list[num][0]
                feature_names.append(name)
                

            # get feature importance ------------------------------------------
            feature_importance = [n for n in pipe[1].coef_[0].tolist()]
            
            
            
            # sort the feature list so it will be sorted in descending order 
            sort_list = sorted(zip(feature_names,feature_importance),reverse=True, key = lambda x: x[1])
            df_feature_ranking = pd.DataFrame(sort_list, columns=['Feature','Coefficient'])
            
            
            print('selected feature numbers:', len(sort_list))
            print(df_feature_ranking)
        
            
            vif = []
            
            for feat in df_feature_ranking.Feature.tolist():
                vif.append(vif_data[vif_data['Feature']== feat]['VIF'].values[0])
                
            df_feature_ranking['VIF'] = vif
            df_feature_ranking.insert(loc=0,column='Context Length', value=[con_len] * len(df_feature_ranking))
        

            # append the accuracy and number of features for plots
            if con_len == 2:
                num_features_two.append(len(vif))
                accu_two.append(accuracy_score(y_test,y_pred))
                
            elif con_len == 5:
                num_features_five.append(len(vif))
                accu_five.append(accuracy_score(y_test,y_pred))
                
            else:
                num_features_ten.append(len(vif))
                accu_ten.append(accuracy_score(y_test,y_pred))
                
            
            
            # save the feature list into csv 
            df_feature_ranking.to_csv("/data/s3619362/vif_feature/" + lang + str(con_len) +".csv")
        
        
        
            
    # plot for accuracy and selected features ---------------------------------
    
    width = 0.25
    #plt.figure(figsize=(200,300))
    fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2,figsize=(10,8)) 
    #fig.subplot(211, facecolor='white')
    ax1.grid(False)
    
    # Set position of bar on X axis 
    x = np.arange(len(lang_pair)) 
    
    
    
    rects1 = ax1.bar(x - width,  accu_two, width, label='2 sent', color='darkturquoise')
    rects2 = ax1.bar(x , accu_five, width, label='5 sent', color='cadetblue') 
    rects3 = ax1.bar(x + width, accu_ten, width, label='10 sent', color='powderblue')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set(ylim=[0.4,0.9])
    ax1.set_ylabel('Accuracy', fontsize=14)
    #ax1.set_title('MT-PE classification VIF',fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lang_pair)
    ax1.set_facecolor('white')
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
                        ha='center', va='bottom', fontsize=12)
        
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
    ax2.set(ylim=[0,20])
    ax2.set_ylabel('Numbers of Selected Features', fontsize=14)
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(lang_pair)
    ax2.set_facecolor('white')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
        
    autolabel(feat1,ax2)
    autolabel(feat2,ax2)
    autolabel(feat3,ax2)
    
    #plt.savefig('/Users/yuwen/Desktop/Thesis/Project/final_plot/vif-htpe.jpg', dpi=300)
    plt.show()
            
            