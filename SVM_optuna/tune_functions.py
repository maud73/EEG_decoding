import optuna
import pandas as pd
import torch 
import os
from sklearn.metrics import balanced_accuracy_score, f1_score

from helpers import resize_batch
from model import SVM_pixel



def find_hyperparam(path_to_save,device, weight_loss,input_size,o_train_loader, o_val_loader, num_epochs, n_trials, num_pixels = 25):
  '''
  Find the hyperparameters using Optuna library

  Args:
      path_to_save (str): path to save the results
      device (torch.device): device to train on
      weight_loss (torch.Tensor): weights to balance the loss function
      input_size (int): size of a input datapoint
      o_train_loader (torch.utils.data.DataLoader): training set used for the hyperparameter searching
      o_val_loader (torch.utils.data.DataLoader): validation set used for the hyperparameter searching
      num_epochs (int): number of epochs
      n_trials (int): number of trials
      num_pixels (int): number of pixels, default = 25

  Returns:
      to_return (pd.DataFrame): hyperparameters
  '''
   
  # Preparing the saving 
  optuna_result = pd.DataFrame(columns = ["Number of finished trials",
                                          "Number of pruned trials",
                                          "Number of complete trials",
                                          "Best accuracy",
                                          "Param"])

  to_return = pd.DataFrame(columns = ['lr', 'beta1', 'beta2', 'weight_decay', 'reg_term', 'scheduler'])


  for i in range(num_pixels):
    print('Pixel n°',i)
    weight = weight_loss[i].to(device)

    # Run a tuning for each pixels model
    df = run_optuna(i, weight, device, input_size, o_train_loader, o_val_loader, num_epochs, n_trials)

    # Saving the results of the trials
    optuna_result.loc[len(optuna_result.index)] =  [df["Number of finished trials"],
                                                    df["Number of pruned trials"],
                                                    df["Number of complete trials"],
                                                    df["Best accuracy"],
                                                    df["Param"]]
    
    if df['Param']['scheduler'] == 'CosineAnnealingLR' : 
       scheduler_dict = {'scheduler' : df['Param']['scheduler'], 'eta_min' : df['Param']['eta_min']}
    else :
       scheduler_dict = {'scheduler' : df['Param']['scheduler'], 'gamma' : df['Param']['gamma'],'step_size': df['Param']['step_size']}
    
    # Saving the best parameters
    to_return.loc[len(to_return.index)] = [df["Param"]['lr'],
                                           df["Param"]["beta1"],
                                           df["Param"]["beta2"],
                                           df["Param"]['weight_decay'],
                                           df["Param"]['reg_term'],
                                           scheduler_dict]

  # Saving 
  os.makedirs(path_to_save, exist_ok=True)

  optuna_result.to_csv(path_to_save + '/optuna_running.csv')
  to_return.to_csv(path_to_save + '/hyperparam.csv')

  print('hyperparameters saved in /trials directory')

  return to_return

def run_optuna(num_pixel, weight_, device, input_size, o_train_loader, o_val_loader, num_epochs, n_trials):
    '''
    Run the optuna hyperparameter tunning

    Args:
        num_pixel (int): pixel number
        weight_ (torch.Tensor): weight for the loss function
        device (torch.device): device to train on
        input_size (int): input size
        o_train_loader (torch.utils.data.DataLoader): training set
        o_val_loader (torch.utils.data.DataLoader): validation set
        num_epochs (int): number of epochs
        n_trials (int): number of trials

    Returns:
        optuna_running (dict): optuna results
    '''
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, num_pixel, weight_, device, input_size, o_train_loader, o_val_loader, num_epochs=num_epochs), n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    trial = study.best_trial

    param = {}

    for key, value in trial.params.items():
      param[key] = value

    optuna_running = {"Number of finished trials" :len(study.trials),
              "Number of pruned trials": len(pruned_trials),
              "Number of complete trials": len(complete_trials),
              "Best accuracy": trial.value,
              "Param": param }


    return optuna_running

def objective(trial, num_pixel, weight_, device, input_size, o_train_loader, o_val_loader, num_epochs):
    '''
    Objective function for the optuna hyperparameter tunning

    Args:
        trial (optuna.trial.Trial): trial
        num_pixel (int): pixel number
        weight_ (torch.Tensor): weight for the loss function
        device (torch.device): device to train on
        input_size (int): size of a input datapoint
        o_train_loader (torch.utils.data.DataLoader): training set used for the hyperparameter searching
        o_val_loader (torch.utils.data.DataLoader): validation set used for the hyperparameter searching
        num_epochs (int): number of epochs

    Returns:
        ACC_ (float): balanced accuracy
    '''
    # Optimizer hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 1, log=False)
    beta2 = trial.suggest_float("beta2", 0.9, 1, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    # Regularization hyperparameters
    reg_term = trial.suggest_float('reg_term', 1e-5, 1e-1, log=True)

    # Pseudo model definiton for the study
    model = SVM_pixel(input_size).to(device)

    # Optimaser and criterion for the study
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=weight_decay)
    criterion = torch.nn.MultiMarginLoss(weight = weight_.double())
    
    # Scheduler hyperparameter
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR'])
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_float('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        eta_min = trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(len(o_train_loader.dataset) * num_epochs) // o_train_loader.batch_size,
            eta_min=eta_min
        )

    # Training of the model
    for epoch in range(num_epochs):
      model.train()
      for batch_x, batch_y in o_train_loader:
        batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass 
        output = model(batch_x)

        loss = criterion(output, batch_y.squeeze())

        # Add regularization
        weight = model.fc.weight.squeeze()
        loss += reg_term * torch.sum(weight * weight)

        # Step and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

      # Validation of the model
      model.eval()
      
      # To report 
      with torch.no_grad():
        for batch_x, batch_y in  o_val_loader:
          batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)
          batch_x, batch_y = batch_x.to(device), batch_y.to(device)
          
          # Forward pass 
          output = model(batch_x)

          # Prediction 
          pred = model.predict_label(output)

          # Concatenate the predictions and the labels 
          if 'Y' in locals(): Y= torch.cat((Y, batch_y))
          else : Y = batch_y 
        
          if 'PRED' in locals() : PRED = torch.cat((PRED,pred))
          else : PRED = pred

      ACC_ = balanced_accuracy_score(Y.flatten().cpu(), PRED.flatten().cpu())

      trial.report(ACC_, epoch)

      # Handle pruning based on the intermediate value
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return ACC_

