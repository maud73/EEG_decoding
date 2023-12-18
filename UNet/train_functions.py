import torch
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

def predict(output):
    """
    Predicts each pixel's class (i.e. the decoded visual stimulus) from the UNet's output

    Args:
        output (Tensor): n_batch, depth 2, height and width 5 (output of the U-Net)

    Returns:
        pred (Tensor): n_batch, height and width 5
    """
    pred = torch.argmax(output,dim=1)
    return pred

def accuracy(prediction, target):
    """
    Computes the batch accuracy of the prediction

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): target stimulus (n_batch, height and width 5)

    Returns:
        accuracy (float)
    """
    N = prediction.shape[0]
    temp = torch.all(torch.all(torch.eq(prediction, target), dim=-1), dim=-1) # Dimension n_batch
    acc = torch.sum(temp)
    return (acc / N).cpu()

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.WeightedFocalLoss
    @param dataset_train: torch.utils.data.DataLoader
    @param dataset_test: torch.utils.data.DataLoader
    @param optimizer: torch.optim.Optimizer (AdamW)
    @param scheduler: torch.optim.lr_scheduler (CosineAnnealingLR(optimizer, T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,)
    @device:
    """
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


def run_training(
    model,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    train_loader,
    device="cuda"
):

    # ===== Model =====
    model = model.to(device=device)

    # ===== Train Model =====
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
