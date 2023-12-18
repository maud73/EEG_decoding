from SVM import *
from Data_processing import get_dataloaders, get_data
from reproducibility import set_random_seeds

def main() :
  print("SVM")
  # === Set random seeds for reproducibility ===
  set_random_seeds()
  
  # === Parameters ===
  test_size = 0.2

  num_epochs = 6
  batch_size = 64

  # === Data ===
  file_path = 'data/resampled_epochs_subj_0.pkl'
  path_to_save = 'trials'

  epochs, labels = get_data(file_path, convention_neg=True)
  train_loader, test_loader = get_dataloaders(epochs[:155], labels[:155], batch_size, test_size, return_val_set=False)
  
  # === Find the ratio that caracterize the balancy between the class ===
  _ , weights = balance_weight(labels)

  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # === Hyperparameters ===

  hyperparam = pd.read_csv(path_to_save + '/hyperparam.csv')

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
  SVMmodel = SVM(device, input_size, pixel_nb = 25)
  SVMmodel = SVMmodel.to(device)

  #Train and validate the model
  Training_results = train(SVMmodel,train_loader, device, weights,num_epochs, hyperparam) #columns=['Pixel n°', 'Training loss','Learning rate history', 'Training accuracy', 'Validating accuracy']

  #Test the model
  print("Testing ...")
  Testing_results = test(SVMmodel, test_loader, path_to_save,weights, device) #columns=['Testing singles F1', 'Testing singles accuracy', 'Testing full F1', 'Testing full acc']

  # ===== Saving and Plots =====
  #save
  print("Plot and save ...")
  save_trial(Training_results, Testing_results, param, path_to_save)

  #plot and save 
  plot_training(Training_results, num_epochs, path_to_save)
  plot_testing(Testing_results, path_to_save)


if __name__ == "__main__":
    main()
