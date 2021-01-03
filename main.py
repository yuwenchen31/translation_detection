#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:56:34 2020

@author: yuwen
"""

# This file conduct the classification for mtht, htpe, mtpe
# It conducts the following steps: 
# 1. Grid search on training data to get the best hyperparameters 
# 2. Use the best hyperparameters to conduct recursive feature elimination (RFECV) 
#    to find the best number of features 
# 3. Use the best features to predict on the testing set

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
import nltk
nltk.download('punkt')

from sklearn.model_selection import StratifiedKFold
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import classification_report,accuracy_score

from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
import sys



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
        # new_df = data_frame.values.reshape(-1,1)
        # print(new_df.shape)
        return data_frame.values.reshape(-1,1)
    

# extract each feature and standarize the data ---------------------------------
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



if __name__ == "__main__":

    
    # clf types: mtht, htpe, mtpe
    clf_type = sys.argv[1]
    
    # save the data fro plotting the figures  
    num_features_two = []
    num_features_five = []
    num_features_ten = []
    
    accu_two = []
    accu_five = []
    accu_ten = []


    context_length = [2,5,10]
    
    # mtht - 11 language pairs 
    if clf_type == 'mtht':
        lang_pair = ['ende', 'enfi', 'enlt', 'enru','deen','fien','guen','kken','lten','ruen','zhen']

    # htpe (zhen) - compares data in htpe and mtht 
    elif clf_type == 'htpe':
        lang_pair = ['htpe','mtht']    

    # mtpe - 5 language pairs 
    else:
        lang_pair = ['ende','enru','enfr','ennl','enpt']
    
    # first go through each language pairs 
    for lang in lang_pair:
        
       
        # create a dataframe to store final features and their importance (coefficients)
        df_feature_ranking = pd.DataFrame()
        
        # for each language pairs, go through each context length 
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
            test_file = "/data/s3619362/mtht/" + test_file
            
        
            train = pd.read_pickle(train_file)
            dev = pd.read_pickle(dev_file)
            test = pd.read_pickle(test_file)
            
            # eliminate the label column
            train_column = list(train.columns)
            train_column.remove('LABEL')
            dev_column = list(dev.columns)
            dev_column.remove('LABEL')
            test_column = list(test.columns)
            test_column.remove('LABEL')
            
            # get the X(instances) and y(labels)
            X_train = train[train_column]
            y_train = train[['LABEL']].values.ravel()
            X_dev = dev[dev_column]
            y_dev = dev[['LABEL']].values.ravel()
            X_test = test[test_column]
            y_test = test[['LABEL']].values.ravel()
            

            #binarize the labels 
            le = preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_dev = le.fit_transform(y_dev)
            y_test = le.fit_transform(y_test)
            
            
            # concatenate the original test and dev file as final testing set
            # Final testing set is shown as follows: 
            # mtht - LP-prev: newstest2019, LP-2019: final 30% of the newstest2019
            # htpe - final 30% of the newstest2017
            # mtpe - varies (please check section 3.3 in the thesis)
            X_test = pd.concat([X_test,X_dev],ignore_index=True)
            y_test = np.concatenate((y_test, y_dev), axis=0)
            

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
                
                'pronoun ratio': pronoun_ratio,
                'passive verb ratio': passive_verb_ratio,
                
        
                }
          
            # get all features names from the column names in the dataframe
            all_features = X_train.columns.tolist()
           
                
            # transfromer list for Feature Union: ['feature name', feature functions]
            union_list = [(k, v) for k, v in feature_mapping.items() if k in all_features]
            
           
            # feature union
            feats = Pipeline([ 
                
                    ('union',FeatureUnion(union_list)
                     
                    )])
            
            
            # parameter grids: hyperparaemters used in grid search 
            parameters = {
                    
                    
                      'clf__C': [2,5,100,1000],
                      'clf__loss': ['hinge', 'squared_hinge'],
                      'clf__random_state': [42],
                      'clf__class_weight':['balanced', None],
                      'clf__tol': [1e-3, 1e-4],
                      'clf__max_iter':[2000,10000,100000],


                    
                      }
            
            
            # initialize the classifier 
            clf = LinearSVC()
        

            # pass all features to the classifier using pipeline
            pipe = Pipeline([
                
                        ('features', feats),
                             
                        
                        ('clf',clf)
                        
                 
            
                        ])
            

            # define grid search with 10-fold cross validation
            grid_search = GridSearchCV(pipe, parameters,return_train_score=True, error_score=0.0, cv=StratifiedKFold(10, random_state=42, shuffle=True), scoring='accuracy')
            
            # fit the grid search to the training data
            grid_search.fit(X_train,y_train)
            
            # extract the best hyperparaemters 
            param_dict = {x.replace("clf__", ""): v for x, v in grid_search.best_params_.items()}
            
        
            # use the found best hyperparameters from grid search for the classifier used for rfe
            clf_for_rfe = LinearSVC(**param_dict)
            
            # initialize rfecv with 10-fold cross validation
            rfe = RFECV(estimator=clf_for_rfe, step=1,cv=StratifiedKFold(10, random_state=42, shuffle=True))            
            
            # pass the features to the rfe classifier by pipelne
            rfe_pipe = Pipeline([
                
                        ('features', feats),

                        ('rfe', rfe),

            
                        ])
            
            
            # fit rfe classifer on training data
            rfe_pipe.fit(X_train,y_train)
            
            # plot the cross-validation score with different # of selected features 
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score")
            plt.title(lang.upper())
            plt.plot(range(1, len(rfe_pipe.named_steps.rfe.grid_scores_) + 1), rfe_pipe.named_steps.rfe.grid_scores_)
            plt.savefig("/home/s3619362/thesis/mtht/test/"+lang+str(con_len), dpi=300)
            plt.show()
            
            
            # predict on unseen data 
            y_pred = rfe_pipe.predict(X_test)
            
            # get the selected features 
            support = rfe_pipe.named_steps.rfe.support_.tolist()

            # number of selected features 
            true_features = Counter(support)[True]
            

            # get feature names -----------------------------------------------
            feature_num = len(pipe.named_steps.features.named_steps.union.transformer_list)
            feature_names = []
            
            # extract all feature names 
            for i in range(feature_num):
                name = pipe.named_steps.features.named_steps.union.transformer_list[i][0]
                feature_names.append(name)
                
            # get selected feature names 
            all_selected_features = [x for x,y in zip(feature_names,support) if y == True]
            print('final selected features are:', all_selected_features)


            
            # get feature importance (coefficients) ------------------------------------------
            feature_importance = [n for n in rfe_pipe.named_steps.rfe.estimator_.coef_[0].tolist()]
            
            
            # sort the feature list so it will be sorted in descending order 
            sort_list = sorted(zip(all_selected_features,feature_importance),reverse=True, key = lambda x: x[1])

            
            # save the features and coefficients as csv file 
            df_feature_ranking = pd.DataFrame(sort_list, columns=['Feature Name','Coefficient'])
            df_feature_ranking.insert(loc=0,column='Context Length', value=[con_len] * len(df_feature_ranking))
            df_feature_ranking.to_csv('/home/s3619362/selected_features/mtht/' + lang + str(con_len) + '.csv')
        

            # append the accuracy and number of selected features for plotting  
            if con_len == 2:
                num_features_two.append(len(feature_importance))
                accu_two.append(accuracy_score(y_test,y_pred))
                
            elif con_len == 5:
                num_features_five.append(len(feature_importance))
                accu_five.append(accuracy_score(y_test,y_pred))
                
            else:
                num_features_ten.append(len(feature_importance))
                accu_ten.append(accuracy_score(y_test,y_pred))
            

    # plot for accuracy and selected features ---------------------------------
    
    width = 0.25
    fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2,figsize=(12,8)) 
    ax1.grid(False)
    
    # Set position of bar on X axis 
    x = np.arange(len(lang_pair)) 

    
    rects1 = ax1.bar(x - width,  accu_two, width, label='2 sent', color='darkturquoise')
    rects2 = ax1.bar(x , accu_five, width, label='5 sent', color='cadetblue') 
    rects3 = ax1.bar(x + width, accu_ten, width, label='10 sent', color='powderblue')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set(ylim=[0.4,0.9])
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('MT-HT classification results on test',fontsize=14)
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
                        ha='center', va='bottom', fontsize=7)
        
    autolabel(rects1,ax1)
    autolabel(rects2,ax1)
    autolabel(rects3,ax1)
    
    fig.tight_layout()
    
    import seaborn as sns
    
    ax2.grid(False)
    
    feat1 = ax2.bar(x - width, num_features_two, width, label='2 sent', color='darkturquoise')
    feat2 = ax2.bar(x , num_features_five, width, label='5 sent', color='cadetblue')
    feat3 = ax2.bar(x + width, num_features_ten, width, label='10 sent', color='powderblue')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set(ylim=[0,30])
    ax2.set_ylabel('Selected Feature Numbers', fontsize=14)
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(lang_pair)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
        
    autolabel(feat1,ax2)
    autolabel(feat2,ax2)
    autolabel(feat3,ax2)
    plt.show()
            
           