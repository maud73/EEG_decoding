from SVM import *
from SVM.Data_processing import get_dataloaders, get_data, get_valset, get_optuna_dataloaders

def main() :
  val_size = 0.3
  optuna_val_size = 0.3
  test_size = 0.3
  batch_size = 16

  file_path = 'resampled_epochs_subj_0.pkl'
  path_to_save = 'trials'
  
  epochs, labels = get_data(file_path)
  train_loader, _ = get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=False)
  optuna_dataset = get_valset(train_loader,val_size)
  o_train_loader, o_val_loader = get_optuna_dataloaders(optuna_dataset, batch_size, optuna_val_size)

  #find the ratio that caracterize the balancy between the class 
  _, weight_loss = balance_weight(labels)

  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #===== HyperParameters =====
  print("Hyperparameters searching ...")
  hyperparam = find_hyperparam(path_to_save,device, weight_loss,input_size, o_train_loader, o_val_loader)

  print('hyeprarameters:', hyperparam)

if __name__ == "__main__":
    print('its working')
    main()


