import torch

def resize_batch(batch_x, batch_y, num_pixel):
  '''
  Resize the batch to fit the model

  Args:
      batch_x (torch.Tensor): batch of data
      batch_y (torch.Tensor): batch of labels
      num_pixel (int): pixel number

  Returns:
      batch_x (torch.Tensor): resized batch of data
      batch_y (torch.Tensor): resized batch of labels
  '''

  # Reshape x_batch
  batch_x= batch_x.flatten(2)
  shape0 = batch_x.shape[0]
  shape1 = batch_x.shape[2]
  batch_x = batch_x.reshape([shape0, shape1])

  # Flatten the labels
  batch_y = batch_y.flatten(1)

  # Take the right pixel
  batch_y = batch_y[:,num_pixel]
  batch_y = batch_y.reshape(batch_y.shape[0], 1)

  # y label should be int64
  batch_y= batch_y.to(torch.int64)

  return batch_x, batch_y

def balance_weight(labels):
  '''
  Compute the weight for the loss function. The weights are pixel specific and correspond to the frequence of each label class

  Args:
      labels (torch.Tensor): labels

  Returns:
      item (torch.Tensor): labels [0, 1]
      alphas (torch.Tensor): weights
  '''

  labs = labels.flatten(1)

  alphas = []
  for i in range (labs.shape[1]) :
    lab = labs[:,i]
    item, count = torch.unique(lab, return_counts=True)
    alpha = count/lab.shape[0]
    alphas.append(1-alpha)

  return item, alphas

