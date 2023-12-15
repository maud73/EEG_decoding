from SVM import *
from SVM.Data_processing import get_dataloaders, get_data

def main() :
  # ===== Data =====
  file_path = 'resampled_epochs_subj_0.pkl'
  path_to_save = 'trials'

  test_size = 0.3  

  epochs, labels = get_data(file_path)
  train_loader, test_loader = get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=False)
  
  #find the ratio that caracterize the balancy between the class 
  item, weight_loss = balance_weight(labels)

  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #===== HyperParameters =====

  hyperparam = pd.read_csv('path_to_save/hyperparam.csv')

  param = {'num_epochs': num_epochs,
           'batch_size': batch_size,
           'Optimizer param lr': hyperparam['lr'],
           'Optimizer param batas' : (hyperparam['beta1'], hyperparam['beta2']),
           'Optimizer param eps' : 1e-08,
           'Optimizer param weight decay' : hyperparam['weight_decay'],
           'scheduler param step size' : hyperparam['step_size'],
           'scheduler param gamma': hyperparam['gamma']}

  # ===== Model =====
  #build a model
  print('Building the model...')
  SMVmodel = SVM(device, input_size, pixel_nb = 25)
  SMVmodel = SMVmodel.to(device)

  #Train and validate the model
  print("Training ...")
  Training_results = train(SMVmodel,train_loader, device, weight_loss, hyperparam) #columns=['Pixel n°', 'Training loss','Learning rate history', 'Training accuracy', 'Validating accuracy']

  #Test the model
  print("Testing ...")
  Testing_results = test(SMVmodel, test_loader, path_to_save, device) #columns=['Testing singles F1', 'Testing singles accuracy', 'Testing full F1', 'Testing full acc']

  # ===== Saving and Plots =====
  #save
  print("Plot and save ...")
  save_trial(Training_results, Testing_results, param, path_to_save)

  #plot and save 
  plot_training(Training_results, num_epochs, path_to_save)
  plot_testing(Testing_results, path_to_save)


if __name__ == "__main__":
    main()
