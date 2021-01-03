# Automatic detection of different types of translation based on translationese features

We employed a Support Vector Machines (SVMs) classifier and conducted three binary classification tasks (i.e., machine translation (MT) vs. human translation (HT), MT vs. post-edit (PE), and HT vs. PE) using the linguistic attributes inspired by translation studies. Those linguistic features model the phenomenon of translationese in four aspects - simplification, explicitation, normalization, and interference. For details, please see the complete version of my [master thesis](https://drive.google.com/file/d/1Xlr-K9PZ7IhBoB4Bm-k7ghXkvOegMauC/view?usp=sharing). 

### Install Dependencies

To install all the packages used in this project, please install the dependencies by

```
pip install -r requirements.txt
```


### Command Line Arguments 

To reproduce the results, execute the `main.py` with command line options of `mtht`, `htpe`,`mtpe`, depending on which classification task you are intended to reproduce.

```
python main.py mtht
```

This also works for classification task using features with low (<5) Variance Inflation Factor (VIF).

```
python vif_classification.py mtht
```

### Files and Directories

- `main.py` - implements the classification with grid search and recursive feature elimination (RFECV). (section 4.2)
- `vif_classification.py` -  implements the classification using features whose VIF are below 5. (section 4.2)
- `all_feat_classification.py` - implements the classification all features (including PoS and character ngrams). Please note that it uses different data format than other two classification, which can be downloded [here](https://drive.google.com/drive/folders/1UITeVlCnln1Lh4etldm30klvAe3NcL4_?usp=sharing).
- `data/` - pickle files. Each column is the feature value. 
- `preprocessing/` - contains files for extracting the non-translationese parts from the raw data (newstest) and splitting data (section 3)
   - `mtht_data.py` - extracts the non-translationese from [newstest2016-2019](http://www.statmt.org/wmt19/results.html).
   - `htpe_data.py` - extracts the ht and pe part from [Microsoft Human-Parity](https://github.com/MicrosoftTranslator/Translator-HumanParityData). 
   - `mtpe_data.py` - extracts mt and pe from [Automatic Post-Editing (APE) shared task](http://www.statmt.org/wmt19/ape-task.html) and [APE-QUEST](https://ape-quest.eu/downloads/).
   - `train_dev_test_split.py` - split data into 70% of training and 30% of testing
- `coefficient_calculation/` - files to extract the feature importance (section 5.2)
   - `1_combine_all.py` - combines different language pairs and context length into one csv file
   - `2_feature_ranking_by_abs_coef.py` - outputs the rankings based on absolute coefficients 
   - `3_coef_rank_table.py` - computes 1) the average coefficients and 2) average rankings per feature 
- `util` - some common functions 
- `feature_calculation/` - files implement the translationese features (section 4.1)
- `selected_features/` - seleccted features after RFECV in `main.py`
- `vif_features/` - features derived after implementing `vif_classification.py`
- `hyperparameter/` - best hyperparameters found by grid search in `main.py`
- `feature_importance/` - outputs derived from `coefficient_calculations`

### Results
The accuracy of each task derived from `main.py` (i.e., PoS and Character ngrams are **not** included in the features) with the context length of 10 consecutive sentences. For the results using the context length of 2 and 5, please see the section 5.1 in the thesis. 

#### MT-HT 
Language Pair  | ende | enfi | enlt | enru | deen | fien | guen | kken | lten | ruen | zhen |
-------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
Accuracy       | 0.51 | 0.62 | 0.6 | 0.55 | 0.5 | 0.56 | 0.76 | 0.67 | 0.58 | 0.6 | 0.57 |          
 
#### HT-PE 

Language Pair  | zhen |
-------------- | ---- |
Accuracy  | 0.72 |

#### MT-PE
Language Pair  | ende | enru | enfr | ennl | enpt |
-------------- | ---- | ---- | ---- | ---- | ---- |
Accuracy       | 0.55 | 0.48 | 0.62 | 0.61 | 0.54 |
