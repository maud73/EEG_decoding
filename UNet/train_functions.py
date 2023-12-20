import torch
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

def predict(output):
    '''
    Predict the stimulus from the output of the model, without matching to existing stimulus

    Args:
        output (Tensor): output of the model (n_batch, height and width 5)

    Returns:
        pred (Tensor): predicted stimulus (n_batch, height and width 5)
    '''
    pred = torch.argmax(output,dim=1)
    return pred

def accuracy(prediction, target):
    '''
    Compute the hard accuracy of the model, as opposed to a pixel-wise accuracy

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): true stimulus (n_batch, height and width 5)

    Returns:
        acc (float): accuracy
    '''
    N = prediction.shape[0]
    temp = torch.all(torch.all(torch.eq(prediction, target), dim=-1), dim=-1) # Dimension n_batch
    acc = torch.sum(temp)
    return (acc / N).cpu()

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    '''
    Train the model for one epoch

    Args:
        model (torch.nn.Module): UNet model to train
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        criterion (torch.nn.Module): loss function
        train_loader (torch.utils.data.DataLoader): training set
        epoch (int): current epoch
        device (str): device to use, 'cuda' or 'cpu'

    Returns:
        loss_history (list): loss history of the epoch
        accuracy_history (list): accuracy history of the epoch
        soft_accuracy_history (list): balanced accuracy history of the epoch
        f1_history (list): F1 score history of the epoch
        lr_history (list): learning rate history of the epoch
    '''
    model.train()
    loss_history = []
    accuracy_history = []
    soft_accuracy_history = []
    f1_history = []
    lr_history = []

    for batch_idx, (data,target) in enumerate(train_loader) :

      data = data.to(device)
      target = target.to(device)

      # Forward pass
      optimizer.zero_grad()
      output = model(data)

      # Compute the gradient
      loss = criterion(output,target)
      loss.backward()

      # Update the parameters of the model with a gradient step
      optimizer.step()
      scheduler.step()

      pred = predict(output)
      acc = accuracy(pred, target)
      soft_acc = balanced_accuracy_score(target.view(-1).cpu(), pred.view(-1).cpu())
      f1 = f1_score(target.view(-1).cpu(), pred.view(-1).cpu())

      loss_float = loss.item()
      loss_history.append(loss_float)
      accuracy_history.append(acc)
      soft_accuracy_history.append(soft_acc)
      f1_history.append(f1)

      lr_history.append(scheduler.get_last_lr()[0])
        
      # Keep track of the training metrics
      if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
          print(
              f"Train Epoch: {epoch}-{batch_idx:03d} "
              f"batch_loss={loss_float:0.2e} "
              f"batch_acc={acc:0.3f} "
              f"batch_soft_acc={soft_acc:0.3f} "
              f"batch_f1={f1:0.3f} "
              f"lr={scheduler.get_last_lr()[0]:0.3e} "
          )

    return loss_history, accuracy_history, soft_accuracy_history, f1_history, lr_history


def run_training(model, optimizer, scheduler, criterion, num_epochs, train_loader, device="cuda"):
    '''   
    Train the model

    Args:
        model (torch.nn.Module): UNet model to train
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        criterion (torch.nn.Module): loss function
        num_epochs (int): number of epochs
        train_loader (torch.utils.data.DataLoader): training set
        device (str): device to use, 'cuda' or 'cpu'

    Returns:
        train_acc (float): accuracy on the training set
        lr_history (list): learning rate history
        train_loss_history (list): loss history
        train_acc_history (list): accuracy history
        train_soft_acc_history (list): balanced accuracy history
        train_f1_history (list): F1 score history
    '''
    model = model.to(device=device)

    # Train
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    train_soft_acc_history = []
    train_f1_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_soft_acc, train_f1, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        train_soft_acc_history.extend(train_soft_acc)
        train_f1_history.extend(train_f1)
        lr_history.extend(lrs)

    return sum(train_acc) / len(train_acc), lr_history, train_loss_history, train_acc_history, train_soft_acc_history, train_f1_history
