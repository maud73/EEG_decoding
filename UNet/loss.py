import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def label_weights(target):
    N = target.shape[0]*25
    freq_min = torch.sum(target == 1) / N
    weights = torch.where(target==0, freq_min.float(), (1-freq_min).float())
    return weights


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
