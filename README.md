# Automatic detection of different types of translation based on translationese features

We employed a Support Vector Machines (SVMs) classifier and conducted three binary classification tasks (i.e., MT-HT, ME-PE, and HT-PE) using the linguistic attributes inspired by translation studies. Those linguistic features model the phenomenon of translationese in four aspects - simplification, explicitation, normalization, and interference. For details, please see the complete version of my [master thesis](https://drive.google.com/file/d/1Xlr-K9PZ7IhBoB4Bm-k7ghXkvOegMauC/view?usp=sharing). 

### Command Line Arguments 

To reproduce the results, execute the `main.py` with command line options of `mtht`, `htpe`,`mtpe`, depending on which classification task you are intended to reproduce.

```
python main.py mtht
```

### Files and Directories

- `main.py` - implements the classification with grid search and recursive feature elimination (RFECV). 
- `vif_classification.py` -  implements the classification using features whose Variance Inflation Factor (VIF) are below 5. 
- `feature_calculation/` - 


### Results 
