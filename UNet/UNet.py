import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from sklearn.metrics import f1_score

# Code edited from https://github.com/amirhosseinh77/UNet-AerialSegmentation/blob/main/model.py

class DoubleConv(nn.Module):
    """Consists of Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='valid'):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
      return self.double_conv(x)


class Down(nn.Module):
    """Consists of MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Consists of transpose convolution then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x1, x2):
        """x1, x2 (Tensor) : n_batch, channels, height, width"""

        # Crop x2 tensor to match the height and width of x1
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Consists of transpose convolution then double conv"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes): # n_classes should be 2, because of binary classification : foreground and background class (gray or black)
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64, kernel_size=3)
        self.down1 = Down(64, 128, kernel_size=3)
        self.down2 = Down(128, 256, kernel_size=3)
        self.down3 = Down(256, 512, kernel_size=2)
        self.down4 = Down(512, 1024, kernel_size=2) # Kernel size of 2 to fit in (height, width)

        self.up1 = Up(1024, 512, kernel_size=2)
        self.up2 = Up(512, 256, kernel_size=2)
        self.up3 = Up(256, 128, kernel_size=3)
        self.up4 = Up(128, 64, kernel_size=(12,57), stride=(4,2))
        self.outc = OutConv(64, n_classes)

        self.apply(he_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) # 2 channels containing the probabilities of gray and black
        return logits
    
    # Code edited from https://github.com/amirhosseinh77/UNet-AerialSegmentation/blob/main/losses.py

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input, target):
        alpha = label_weights(target).view(-1,1)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        logpt = logpt * Variable(alpha)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'mean':
          return loss.mean()
        elif self.reduction == 'sum':
          return loss.sum()
        elif self.reduction == 'none':
          return loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

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

def soft_accuracy(prediction,target):
    """
    Computes the weighted pixel-wise accuracy of the prediction

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): target stimulus (n_batch, height and width 5)

    Returns:
        pixel-wise accuracy (float)
    """
    weights = label_weights(target)
    acc = weights*(prediction==target)
    return torch.mean(acc).float()

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
      ## DEBUG

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
        #loss = loss.cpu().numpy()
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points

def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    losses = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        #print(data.shape)
        #print(target.shape)
        output = model(data)
        #print(output.shape)

        loss = criterion(output, target)
        #print(loss.shape)
        pred = predict(output)
        #print(pred.shape)
        point = {'target': target.detach().cpu().numpy(),
         'predict': pred.detach().cpu().numpy(),
         'loss': loss.detach().cpu().numpy()
         }
        #print(point)
        losses.append(loss.detach().cpu().numpy())

        points.append(point)

    losses = np.array(losses)

    return points,losses

        #point =
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