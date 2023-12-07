from SVM_model.SVM import *

def main() :
  param = {'num_epochs': num_epochs,
           'batch_size_trainning': batch_size_trainning,
           'Optimizer param lr': lr,
           'Optimizer param batas' : betas,
           'Optimizer param eps' : eps,
           'Optimizer param weight decay' : weight_decay,
           'scheduler param step size' : step_size,
           'scheduler param gamma': gamma}


  # ===== Data =====
  #file_path = 'drive/MyDrive/Project2/resampled_epochs_subj_0.pkl' #for the drive
  file_path = '/EEG_DECODING/data/resampled_epochs_subj_0.pkl'

  epochs, labels = get_data(file_path, convention_neg=True, two_channels=False)
  train_loader, val_loader, test_loader = get_dataloaders(epochs, labels, batch_size=batch_size_trainning)

  input_size = train_loader.dataset[:][0].shape[2]*train_loader.dataset[:][0].shape[0]

  #===== Device =====
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # ===== Model =====
  #build a model
  model = SVM(input_size, num_classes = 60, pixel_nb = 25)
  model.to(device)

  #Train and validate the model
  Training_results = train(model,train_loader,val_loader) #columns=['Pixel n°', 'Trainning loss','Learning rate history', 'Trainning accuracy', 'Validating accuracy']

  #Test the model
  Testing_results = test(model, test_loader) #columns=['Testing singles F1', 'Testing singles accuracy', 'Testing full F1', 'Testing full acc']

  # ===== Saving and Plots =====
  #save
  save_trial(Training_results, Testing_results, param)

  #plot
  plot_training(Training_results, num_epochs)
  plot_testing(Testing_results)
