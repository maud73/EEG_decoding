from helpers import balance_weight
from tune_functions import find_hyperparam
from Data_processing import get_dataloaders, get_data, get_valset, get_optuna_dataloaders
import torch
from reproducibility import set_random_seeds

def main() :

    # === Set random seeds for reproducibility ===
    set_random_seeds()

    # Path to save the results
    path_to_save = 'Trials'

    # === Get the data loader ===
    # Parameters for the Loader
    test_size = 0.2
    val_size = 0.3
    optuna_val_size = 0.3

    n_trials = 25
    num_epochs = 30 
    batch_size = 32 

    # Path to th data
    file_path = 'data/resampled_epochs_subj_0.pkl'

    # DataLoaders
    epochs, labels = get_data(file_path, convention_neg=False)
    train_loader, _ = get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=False)
    optuna_dataset = get_valset(train_loader,val_size)
    o_train_loader, o_val_loader = get_optuna_dataloaders(optuna_dataset, batch_size, optuna_val_size)

    # Amout of each labels  
    _, weights = balance_weight(labels)

    input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===== HyperParameters =====
    print("Hyperparameters searching ...")
    hyperparam = find_hyperparam(path_to_save,device, weights,input_size, o_train_loader, o_val_loader, num_epochs, n_trials)

    print('hyeprarameters:', hyperparam)

if __name__ == "__main__":
    main()


