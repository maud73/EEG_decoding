from helpers import balance_weight
from tune_functions import find_hyperparam
from Data_processing import get_dataloaders, get_data, get_valset, get_optuna_dataloaders
import torch
from reproducibility import set_random_seeds

"""
Pipeline for Hyperparameter Optimization Using Optuna.

1. Defining parameters:
    - The seed is fixed for reproducibility
    - Set the path to save the results
    - Set the optuna parameters: number of epochs per trial, the number of trials, the split factor.

2. Get the data:
    - get the preprocess data

3. Hyperprarmeters searching using Optuna:
    - Find the best hyperparameters using a maximisatino of the balanced accuracy.
    - Save the outputs of the trials. 
    - Save the best hymperprameters.
"""

def main() :

    # === Parameters ===
    set_random_seeds()

    # Path to save the results
    path_to_save = 'Trials'

    # Parameters for Optuna
    test_size = 0.2
    val_size = 0.3
    optuna_val_size = 0.3
    n_trials = 25 
    num_epochs = 30 
    batch_size = 32 

    # === Data loader ===
    # Path to th data
    file_path = 'resampled_epochs_subj_0.pkl'

    # Get the data
    epochs, labels = get_data(file_path, convention_neg=False)
    train_loader, _ = get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=False)
    optuna_dataset = get_valset(train_loader,val_size)
    o_train_loader, o_val_loader = get_optuna_dataloaders(optuna_dataset, batch_size, optuna_val_size)

    # Amout of each labels  
    _, weights = balance_weight(labels)

    # Size of the input datapoint 
    input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

    # Device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===== HyperParameters =====
    print("Hyperparameters searching ...")
    hyperparam = find_hyperparam(path_to_save,device, weights,input_size, o_train_loader, o_val_loader, num_epochs, n_trials)

    print('hyeprarameters:', hyperparam)

if __name__ == "__main__":
    main()


