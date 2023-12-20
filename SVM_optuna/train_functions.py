import torch 
import ast
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from helpers import resize_batch, balanced_accuracy_score


def train_single_model(model, train_loader, num_pixel, num_epoch, device, weight_, param):
    '''
    Train a SVM clissifier model for a single pixel

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training set
        num_pixel (int): pixel number
        num_epoch (int): number of epochs
        device (torch.device): device to train on
        weight_ (torch.Tensor): weight for the loss function
        param (dict): hyperparameters from Optuna tunning

    Returns:
        loss_per_epoch (list): loss per epoch
        acc_per_epoch (list): accuracy per epoch
        wacc_per_epoch (list): weighted accuracy per epoch
        f1_per_epoch (list): f1 score per epoch
        lr_history (list): learning rate per epoch
    '''
    # Criterion, Optimizer definiton
    criterion = torch.nn.MultiMarginLoss(weight=weight_.double())

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['lr'],
                                 betas=(param['beta1'], param['beta2']),
                                 eps=1e-08,)
    
    # Scheduler defintion
    scheduler_string = param['scheduler']
    scheduler_dict = ast.literal_eval(scheduler_string)

    if scheduler_dict['scheduler'] == 'StepLR':
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_dict['step_size'], gamma=scheduler_dict['gamma'])

    
    if scheduler_dict['scheduler'] == 'CosineAnnealingLR':
      scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(len(train_loader.dataset) * num_epoch) // train_loader.batch_size,
            eta_min=scheduler_dict['eta_min'])

    #regularization parameters
    reg_term = param['reg_term']

    # Preparing the saving
    loss_per_epoch = []
    acc_per_epoch = []
    lr_history = []
    wacc_per_epoch = []
    f1_per_epoch = []

    for epoch in range(num_epoch):
      running_corrects = 0.0
      running_loss = []
      running_wacc = 0.0
      running_f1 = 0.0

      # Set model to training model
      model.train()

      for batch_x, batch_y in train_loader:
        batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)

        # Running on right device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)

        preds = model.predict_label(outputs)

        loss = criterion(outputs, batch_y.flatten())

        # Add regularization term
        weight = model.fc.weight.squeeze()
        loss += reg_term * torch.sum(weight * weight)

        # Backward and Steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Saving the metrics and loss
        running_loss.append(loss.cpu().item())
        running_corrects += torch.mean((preds == batch_y.data).float()).cpu()
        
        # Balanced accuracy
        wacc = balanced_accuracy_score(batch_y.flatten().cpu(),preds.flatten().cpu()) # Mean wacc on the batch
        running_wacc += wacc

        # F1 score
        f1 = f1_score(batch_y.flatten().cpu(),preds.flatten().cpu()) # Mean f1 on the batch
        running_f1 += f1

      #save the learning rate, loss, accuracy, lr, balanced accuracy and f1 for each epoch
      lr_history.append(scheduler.get_last_lr()[0])
      running_loss = torch.nanmean(torch.tensor(running_loss))
      loss_per_epoch.append(running_loss)
      acc_per_epoch.append(running_corrects / len(train_loader))
      wacc_per_epoch.append(running_wacc / len(train_loader))
      f1_per_epoch.append(running_f1 / len(train_loader))

      # Plot info only every 100 epochs
      if epoch%100 ==0 : 
        print(f'Epoch: {epoch}',
              'Loss: {:.4f}'.format(running_loss),
              'Acc: {:.4f}'.format(running_corrects/ len(train_loader)),
              'weighted Acc: {:.4f}'.format(running_wacc / len(train_loader)),
              'F1: {:.4f}'.format(running_f1 / len(train_loader)))

    return loss_per_epoch, acc_per_epoch, wacc_per_epoch, f1_per_epoch, lr_history

def train(full_model, train_loader, device, num_epoch, weights_, hyperparam):
  '''
  Train the full model, it's training each of the 25 SVM_pixel model

  Args:
      full_model (torch.nn.Module): model to train composed by 25 SVM_pixel model 
      train_loader (torch.utils.data.DataLoader): training set
      device (torch.device): device to train on
      num_epoch (int): number of epochs
      weights_ (torch.Tensor): weight for the loss function
      hyperparam (pd.DataFrame): hyperparameters form optuna tunning

  Returns:
      to_store (pd.DataFrame): training results
  '''

  to_store = pd.DataFrame(columns=['Pixel n°',
                                   'Training loss',
                                   'Learning rate history',
                                   'Training accuracy',
                                   'Training weighted accuracy',
                                   'Training F1'])

  for i, model in enumerate(full_model.models):
    print('Pixel ', i )

    # Take the weight corresponding to the pixel i 
    weight_ = weights_[i].to(device)

    # Calling the train fucniton for the SVM_model i
    loss_per_epoch, acc_per_epoch, wacc_per_epoch, f1_per_epoch, lr_history = train_single_model(model,
                                                                                   train_loader,
                                                                                   i,
                                                                                   num_epoch,
                                                                                   device, 
                                                                                   weight_,
                                                                                   hyperparam.iloc[i])

    # Store the training information
    to_store.loc[len(to_store.index)] =  [i, loss_per_epoch, acc_per_epoch, wacc_per_epoch, f1_per_epoch, lr_history]

  # Save the model
  torch.save(full_model, 'SVMmodel.pth')

  print('the model is saved under SVMmodel.pth!')

  return to_store

