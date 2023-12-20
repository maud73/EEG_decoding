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
- mne <= 1.3.1
- pickle

## Usage 
### Data

Code set for reproducibility. Long runtime
### Google Colab Notebooks

### EPFL Scitas clusters
1. Access the clusters
2. Set up an environment
3. Git pull the code
4. Download the data with scp
5. Write .run files for each modules
Adapt the path file for the environment
6. sbatch and run
Check Squeue, tail -f, 

### General --> precise the time
## SVM
1. Tune the model: tune.py
Obtain hyperparam.csv
2. Train the model: main.py
Obtain .csv files 
3. Plot the results
4. Test the model
## UNet
1. Tune the model: tune.py
Obtain hyperparam.pkl
2. Train the model: main.py
Obtain .pkl files 
3. Plot the results
4. Test the model

## Authors 
Dupont-Roc Maud, Grosjean Barbara, Ingster Abigaïl 
