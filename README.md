# EEG_decoding
## SVM part
library used : 
- mne
- torch
- numpy 
- matplotlib.pyplot
- matplotlib.ticker
- copy
- pandas
- xgboost import XGBClassifier
- os
- optuna

SVM.py contains:
**class**
- class SVM_pixel
- class SVM
  
**train, eval and test functions**
- train_single_model
- eval_single_model
- train
- test_single_model
- test

**help function**
- F1andscore
- resize_batch
- balance_weight

**save and plot functions**
- save_trial
- plot_training
- plot_testing
- save_prediction
- plot_pattern
- heatmap
- annotate_heatmap
