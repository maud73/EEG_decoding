from SVM import *
from Data_processing import get_dataloaders, get_data, get_valset, get_optuna_dataloaders

def main() :
  
  test_size = 0.2
  val_size = 0.3
  optuna_val_size = 0.3

  n_trials = 2
  num_epochs = 2
  batch_size = 8

  file_path = '../data/resampled_epochs_subj_0.pkl'
  path_to_save = 'trials'
  
  epochs, labels = get_data(file_path)
  train_loader, _ = get_dataloaders(epochs[:155], labels[:155], batch_size, test_size, return_val_set=False)
  optuna_dataset = get_valset(train_loader,val_size)
  o_train_loader, o_val_loader = get_optuna_dataloaders(optuna_dataset, batch_size, optuna_val_size)

  # Find the ratio that characterizes the balance between classes 
  _, weight_loss = balance_weight(labels)

  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # === Hyperparameter tuning ===
  print("Hyperparameters searching ...")
  hyperparam = find_hyperparam(path_to_save,device, weight_loss,input_size, o_train_loader, o_val_loader, num_epochs=num_epochs, n_trials=n_trials)

  print('hyeprarameters:', hyperparam)

if __name__ == "__main__":
    print('its working')
    main()


