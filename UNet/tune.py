import torch
import optuna
import os
import pickle
from tune_functions import objective
from Data_processing import get_data, get_dataloaders, get_valset, get_optuna_dataloaders

"""Pipeline for Hyperparameter Optimization Using Optuna.

1. Load Data:
    - Loads and preprocesses EEG data.

2. Optuna Hyperparameter Optimization:
    - Sets Optuna parameters: number of epochs per trial, the number of trials, the split factor.
    - Splits the validation set into pseudo-train and validation sets for Optuna.

3. Optuna Study Execution:
    - Maximizes the objective function using an Optuna study.

4. Results and Storage:
    - Saves the best trial's hyperparameters along with its corresponding F1 score.

Note: the seeds were not fixed prior to hyperparameter tuning. Exact reproducibility may be impaired.
"""

def main():
    # Load the data and set up parameters
    epochs, labels = get_data(file_path='data/resampled_epochs_subj_0.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = 10
    batch_size = 32
    data_kwargs = dict(
        epochs=epochs,
        labels=labels,
        batch_size=batch_size
    )
    
    # Optuna parameters
    n_trials = 20
    print(f"Run with n_trials={n_trials}, num_epochs={num_epochs}")
    test_size = 0.2
    val_size = 0.3
    optuna_val_size = 0.3
    
    train_loader, test_loader = get_dataloaders(**data_kwargs, test_size=test_size, return_val_set=False)
    
    # Split the validation set into pseudo-train and validation sets
    optuna_dataset = get_valset(train_loader, val_size)
    train_loader_hyp, val_loader_hyp = get_optuna_dataloaders(optuna_dataset, batch_size, optuna_val_size)
    print("Start Optuna")
    
    # Build and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, device, train_loader_hyp, val_loader_hyp, num_epochs), n_trials=n_trials)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    # Save the best trial
    trial = study.best_trial
    param = {}
    for key, value in trial.params.items():
      param[key] = value
    best_params = {'Best F1': trial.value, 'Param': param}
    
    file_path = 'UNet/best_hyperparams.pkl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(best_params, fp)

if __name__ == "__main__":
    main()
