import torch
import optuna
import os
import pickle
from tune_functions import objective
from Data_processing import get_data, get_dataloaders, get_valset, get_optuna_dataloaders

def main():
    print("Start main")
    # Load the data
    epochs, labels = get_data(file_path='data/resampled_epochs_subj_0.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = 2
    batch_size = 5
    data_kwargs = dict(
        epochs=epochs[:155],
        labels=labels[:155],
        batch_size=batch_size
    )
    # Optuna parameter
    n_trials = 2
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
    print("Saving hyperparameters")
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
    print("Saved and finished")
if __name__ == "__main__":
    main()
