import torch
import numpy as np
from sklearn.metrics import f1_score


def label_weights(target):
    N = target.shape[0]*25
    freq_min = torch.sum(target == 1) / N
    weights = torch.where(target==0, freq_min.float(), (1-freq_min).float())
    return weights


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
    Computes the accuracy of the prediction

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): target stimulus (n_batch, height and width 5)

    Returns:
        accuracy (float)
    """
    N = prediction.shape[0]
    acc = 0 if not torch.all(torch.eq(prediction, target)) else 1
    return acc / N

def soft_accuracy(prediction, target, reduction='mean'):
    """
    Computes the weighted pixel-wise accuracy of the prediction

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): target stimulus (n_batch, height and width 5)

    Returns:
        pixel-wise accuracy (float)
    """
    N = prediction.shape[0]
    weights = label_weights(target)
    acc = weights*(prediction==target)
    if reduction == 'mean':
        return torch.mean(acc).cpu().float()
    elif reduction == 'none':
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
      soft_acc = soft_accuracy(pred, target)
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


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    f1 = 0
    soft_acc = 0

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += criterion(output, target).item() * len(data)
        pred = predict(output)
        correct += accuracy(pred, target)
        soft_acc += soft_accuracy(pred, target)
        f1 += f1_score(target.view(-1).cpu(), pred.view(-1).cpu())

    f1 /= len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(
        "Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Soft accuracy: {:.0f}%, F1 score: {}".format(
            val_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
            100.0 * soft_acc / len(val_loader.dataset),
            f1,
        )
    )
    return val_loss, correct / len(val_loader.dataset), soft_acc / len(val_loader.dataset), f1


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = predict(output)

        data = np.split(data.cpu().numpy(), len(data))
        print(loss)
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points

@torch.no_grad()
def get_predictions(model, device, test_loader, criterion, num=None):
    model.eval()
    points = []
    losses = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = criterion(output, target)
        pred = predict(output)
        point = {'target': target.detach().cpu().numpy(),
         'predict': pred.detach().cpu().numpy(),
         'loss': loss.detach().cpu().numpy()
         }
        losses.append(loss.detach().cpu().numpy())

        points.append(point)

    losses = np.array(losses)

    return points,losses


def run_training(
    model,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    optimizer_kwargs,
    train_loader,
    val_loader,
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
    val_loss_history = []
    val_acc_history = []
    val_soft_acc_history = []
    val_f1_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_soft_acc, train_f1, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        train_soft_acc_history.extend(train_soft_acc)
        train_f1_history.extend(train_f1)
        lr_history.extend(lrs)

        val_loss, val_acc, val_soft_acc, val_f1 = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_soft_acc_history.append(val_soft_acc)
        val_f1_history.append(val_f1)

    return sum(train_acc) / len(train_acc), val_acc, val_soft_acc, lr_history, train_loss_history, train_acc_history, train_soft_acc_history, train_f1_history, val_loss_history, val_acc_history, val_soft_acc_history, val_f1_history