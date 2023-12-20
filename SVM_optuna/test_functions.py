import torch 
from sklearn.metrics import f1_score, balanced_accuracy_score
import pandas as pd

from helpers import resize_batch
from save_plot_functions import save_prediction

def test_single_model(model, test_loader, num_pixel, device):
  '''
  Test a SVM classifier model for a single pixel

  Args:
      model (torch.nn.Module): SVM_pixel model to test
      test_loader (torch.utils.data.DataLoader): test set
      num_pixel (int): pixel number
      device (torch.device): device to train on

  Returns:
      F1 (float): F1 score
      Wacc (float): balanced accuracy
      acc (float): accuracy
  '''

  running_corrects =0
  running_wacc=0
  running_f1 =0
  with torch.no_grad():
    for batch_x, batch_y in test_loader:

        batch_x, batch_y = resize_batch(batch_x, batch_y, num_pixel)

        #running on the device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)

        preds = model.predict_label(outputs)

        # Save the metrics
        running_corrects += torch.mean((preds == batch_y.data).float()).cpu()
        
        # Balanced accuracy
        wacc = balanced_accuracy_score(batch_y.flatten().cpu(),preds.flatten().cpu())
        running_wacc += wacc

        # F1 score
        f1 = f1_score(batch_y.flatten().cpu(),preds.flatten().cpu())
        running_f1 += f1

    acc = running_corrects.item() / len(test_loader)
    Wacc = running_wacc / len(test_loader)
    F1 = running_f1/len(test_loader)

  return F1, Wacc, acc

def test(trained_model, test_loader, outpath, device): 
  '''
  Test the full model, it's testing each of the SVM_pixel model with the right associated data

  Args:
      trained_model (torch.nn.Module): model to test
      test_loader (torch.utils.data.DataLoader): test set
      outpath (str): path to save the plots
      device (torch.device): device to train on

  Returns:
      to_store (pd.DataFrame): testing results
  '''
  
  to_store = pd.DataFrame(columns=['Testing singles weighted accuracy', 'Testing singles accuracy', 'Testing singles F1'] )

  # Preparing the metrics
  wacc_single = []
  f1_single = []
  acc_single =[]

  for i, model in enumerate(trained_model.models) :

    # Test each of the single model
    f1, wacc, acc = test_single_model(model, test_loader, i, device)
    
    # Saving the metrics 
    f1_single.append(f1)
    wacc_single.append(wacc)
    acc_single.append(acc)

  # Store into a DataFrame
  to_store['Testing singles weighted accuracy'] = wacc_single
  to_store['Testing singles accuracy'] = acc_single
  to_store['Testing singles F1'] = f1_single
  
  k = 0
  with torch.no_grad():
  
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.flatten(2)
        batch_y = batch_y.flatten(1)

        # On the reight device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # One forward of the 25-SVM model 
        outputs = trained_model(batch_x)
        k+=1

        # Save exemple true stimuli vs. predicted one every 100 epochs
        if i%100 ==0 :
          pred_pattern = trained_model.predict_pattern(outputs)

          # Save the prediction under /Trials folder
          save_prediction(batch_y.cpu(), pred_pattern.cpu(), outpath, i) 

  print('saving example pattern into trials/testing_patterns_example')
  return to_store

