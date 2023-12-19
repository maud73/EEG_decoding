# EEG_decoding
*CS-433 Machine learning, EPFL*
### Machine learning for science in collaboration with Translational Neural Engineering Lab

This project aims to decode an electroencephalogram (EEG) recorded in subjects presented with a visual stimulus on a screen, a five-by-five pixel image made of black and gray squares. To this end, we used machine learning models. The first method involved implementing a pixel-wise approach, i.e. training a Support Vector Machine (SVM) model on each pixel for a binary classification task, thereafter grouping each model  to form the whole stimulus. The second approach consisted of using a U-shaped fully convolutional neural network (UNet) to capture the spatial dependency between each recording channel (electrode).

The main classes present are:
- SMV_pixel: Support machine vector binary classifier
- SVM: tweety-five SVM_pixel class lumped
- UNet: a deep neural network based on UNet architecture

## Requirements

- Python >= 3.5
- numpy
- pytorch
- matplotlib
- scikit-learn
- panda
- op
- ast
- optuna
- mne
- pickle

## Autors 
Dupont-Roc Maud, Grosjean Barbara, Ingster Abigaïl 
