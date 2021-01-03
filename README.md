# Automatic detection of different types of translation based on translationese features

We employed a Support Vector Machines (SVMs) classifier and conducted three binary classification tasks (i.e., MT-HT, ME-PE, and HT-PE) using the linguistic attributes inspired by translation studies. Those linguistic features model the phenomenon of translationese in four aspects - simplification, explicitation, normalization, and interference. For details, please see the complete version of my [master thesis](https://drive.google.com/file/d/1Xlr-K9PZ7IhBoB4Bm-k7ghXkvOegMauC/view?usp=sharing). 

### Command Line Arguments 

To reproduce the results, execute the `main.py` with command line options of `mtht`, `htpe`,`mtpe`, depending on which classification task you are intended to reproduce.

```
python main.py mtht
```
This also works for classification task using features with low (<5) Variance Inflation Factor (VIF). 


### Files and Directories

- `main.py` - implements the classification with grid search and recursive feature elimination (RFECV). (section 4.2)
- `vif_classification.py` -  implements the classification using features whose Variance Inflation Factor (VIF) are below 5. (section 4.2)
- `feature_calculation/` - files implement the translationese features (section 4.1)
- `coefficient_calculation/` - files to extract the feature importance metrics (section 5.2)
   - `1_combine_all.py` - 

- `preprocessing/` - contains files for extracting the non-translationese parts from the raw data (newstest) and splitting data (section 3)
  - `mtht_data.py` - extracts the non-translationese from [newstest2016-2019](http://www.statmt.org/wmt19/results.html).
  - `htpe_data.py` - extracts the ht and pe part from [Microsoft Human-Parity](https://github.com/MicrosoftTranslator/Translator-HumanParityData). 
  - `mtpe_data.py` - extracts mt and pe from [Automatic Post-Editing (APE) shared task](http://www.statmt.org/wmt19/ape-task.html) and [APE-QUEST](https://ape-quest.eu/downloads/).
