import torch
import pandas as pd 

from Data_processing import get_dataloaders, get_data
from reproducibility import set_random_seeds

from model import SVM
from helpers import balance_weight
from train_functions import train
from test_functions import test
from save_plot_functions import save_trial, plot_training, plot_testing

"""
This module implements a training and testing pipeline for a full-SVM model, wich corresponds to 25 SVM model.

1. Data Loading:
   - `set_random_seeds()`: Ensures reproducibility by setting random seeds.
   - `get_data(file_path)`: EEG data loading and preprocessing.
   - take the hyperparameter from the file.

2. Model Training and Testing:
   - Trains the full SVM model.
   - Tests the full SVM model.
   - Some results are saved during the testing phase.

3. Saving
   - Save the outputs of the training: Loss, learning rate, accuracy, balanced accuracy and F1 score history.
   - Save the outputs of the testing: F1 score, accuracy, balanced accuracy.
"""

def main() :
  print("SVM")
  # === Data Loading ===
  set_random_seeds()
  
  # Parameters 
  test_size = 0.2

  num_epochs = 100 #for debugs
  batch_size = 64

  # Path 
  file_path = 'resampled_epochs_subj_0.pkl'
  path_to_save = 'Trials'

  epochs, labels = get_data(file_path, convention_neg=False)
  train_loader, test_loader = get_dataloaders(epochs[:155], labels[:155], batch_size, test_size, return_val_set=False) #for debugs
  
  # Find the label ratio
  _ , weights = balance_weight(labels)

  # Size of the input datapoint
  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  # Defining the device to run on
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Hyperparameters 
  hyperparam = pd.read_csv(path_to_save + '/hyperparam.csv')

  param = {'num_epochs': num_epochs,
           'batch_size': batch_size,
           'Optimizer param lr': hyperparam['lr'],
           'Optimizer param batas' : (hyperparam['beta1'], hyperparam['beta2']),
           'Optimizer param eps' : 1e-08,
           'Optimizer param weight decay' : hyperparam['weight_decay'],
           'scheduler' : hyperparam['scheduler']
           }


  # ===== Model =====
  print('Building the model...')
  SVMmodel = SVM(device, input_size, pixel_nb = 25)
  SVMmodel = SVMmodel.to(device)

  # Train and validate the model
  print('training...')
  Training_results = train(SVMmodel,train_loader, device, num_epochs, weights, hyperparam) 

  # Test the model
  print("Testing ...")
  Testing_results = test(SVMmodel, test_loader, path_to_save, device) 

  # ===== Saving =====
  # Save
  print("Saving ...")
  save_trial(Training_results, Testing_results, param, path_to_save)

if __name__ == "__main__":
    main()
