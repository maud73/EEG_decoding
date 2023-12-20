import torch 
from sklearn.metrics import f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np

from helpers import resize_batch, hard_accuracy
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
  with torch.no_grad():
    for batch_x, batch_y in test_loader:

        batch_x, batch_y = resize_batch(batch_x, batch_y, num_pixel)

        #running on the device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)

        preds = model.predict_label(outputs)

        # Concatenate the predictions and the labels 
        if 'Y' in locals():
          Y= torch.cat((Y, batch_y))
        else : Y = batch_y

        if 'PRED' in locals() :
          PRED = torch.cat((PRED,preds))
        else : PRED = preds

        # Save the hard accuracy 
        running_corrects += torch.mean((preds == batch_y.data).float()).cpu()

    acc = running_corrects.item() / len(test_loader)
  
    F1 = f1_score(Y.flatten().cpu(),PRED.flatten().cpu())
    Wacc = balanced_accuracy_score(Y.flatten().cpu(), PRED.flatten().cpu())

  return F1, Wacc, acc

def test(trained_model, test_loader, outpath, device, stimuli): 
  '''
  Test the full model, it's testing each of the SVM_pixel model with the right associated data. Put togathr the single prediction
  fit with the nerest stimuli and selesct 5 of the worst/best prediciton.

  Args:
      trained_model (torch.nn.Module): model to test
      test_loader (torch.utils.data.DataLoader): test set
      outpath (str): path to save the plots
      device (torch.device): device to train on

  Returns:
      to_store (pd.DataFrame): testing results
      points (dict): testing result per point
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
  points = []
  
  with torch.no_grad():
  
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.flatten(2)
        batch_y = batch_y.flatten(1)

        # On the reight device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # One forward of the 25-SVM model 
        outputs = trained_model(batch_x)
        pred_pattern = trained_model.predict_pattern(outputs)
        
        pred_pattern = pred_pattern.cpu()
      
        correlation_coefficients = [np.corrcoef(stimulus.flatten(), pred_pattern.flatten())[0, 1] for stimulus in stimuli]
        closest = np.argmax(correlation_coefficients)
        pred = torch.from_numpy(stimuli[closest]).unsqueeze(0).to(device)

        # Compute the hard accurcy if we match the stimuli
        acc = hard_accuracy(pred_pattern, batch_y)

        f1 = f1_score(batch_y.flatten().cpu(), pred.flatten().cpu())
        soft_acc = balanced_accuracy_score(batch_y.flatten().cpu(), pred.flatten().cpu())

        point = {'target': batch_y.detach().cpu(),
          'predict': pred.detach().cpu(),
            'accuracy': acc.detach().cpu().numpy(),
            'soft_accuracy': soft_acc,
                   'f1': f1,
          }
        
        points.append(point)

    return to_store, points

