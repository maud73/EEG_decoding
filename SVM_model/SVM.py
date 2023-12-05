import torch
import numpy as np
import matplotlib.pyplot as pyplot
import copy
import pandas as pd

#parameters
#for the SVM pixel
num_epochs = 2
batch_size = 3
learning_rate = 0.001 #optimizer
momentum = 0.9 #optimizer
step_size = 10 #scheduler
gamma = 0.1 #scheduler

#class : SVM_pixel and SVM
class SVM_pixel(torch.nn.Module) :
  """ 
  SVM_pixel class, support vector machine model classify single pixel
  into the class -1 or 1. 
  """

  def __init__(self,  input_size, num_classes=2): #dim = len(X)
    super(SVM_pixel, self).__init__()
    self.fc = torch.nn.Linear(input_size, num_classes)
    
    #for the regularisation
    self.reg_type = 'L1' #or 'L2' ?
    self.reg_term = 0.001

  def forward(self, x):
        out = self.fc(x)
        return out

  def predict_label(self, outputs):
    _, preds = torch.max(outputs, 1)
    return preds

class SVM(torch.nn.Module):
  """
  SVM, support machine model for 5x5 image classification.
  """
  def __init__(self,  input_size, num_classes = 60, pixel_nb = 25): #dim = len(X)
    super(SVM, self).__init__()
    self.models = []
    for i in range(pixel_nb) :
      model = SVM_pixel(input_size)
      self.models.append(model)

  def forward(self, x):
    outs= []
    for i,model in enumerate(self.models):
      outs.append(model(x))
    return outs

  def predict_pattern(self, outputs): #output is list of len 25 and element shape are : batch_size, num_label
    for i in range(len(self.models)):
      pred = self.models[i].predict_label(outputs[i])

      if i ==0 : preds = pred
      else : preds = torch.concatenate((preds, pred), axis = 0)

    return preds.reshape(outputs[0].shape[0], len(outputs)) #size batch_size, nb of pixel

#train functions train_single_model and train
def train_single_model(model, train_loader, dataset_sizes, plot = 'False'):
    criterion = torch.nn.SoftMarginLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=momentum)

    scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    #regularization parameters
    reg_type = model.reg_type
    reg_term = model.reg_term

    best_model_wts = copy.deepcopy(model.state_dict())

    loss_per_epoch = []
    metric_per_epoch = []

    for epoch in range(model.num_epochs):
      correct_predict_per_epoch = 0.0
      for batch_x, batch_y in train_loader:
        model.train()

        outputs = model(batch_x)

        preds = model.predict_label(outputs)

        loss = criterion(outputs, batch_y) #mean loss per batch

        # Add regularization:  Full loss = data loss + regularization loss
        weight = model.fc.weight.squeeze()

        if model.reg_type == 'L1':  # add L1 (LASSO - Least Absolute Shrinkage and Selection Operator)
                                                    # loss which leads to sparsity.
            loss += model.reg_term * torch.sum(torch.abs(weight))

        elif model.reg_type == 'L2':   # add L2 (Ridge) loss
            loss += model.reg_term * torch.sum(weight * weight)


        acc = torch.sum(preds == batch_y.data) #accross the batch how many correct we have

        correct_predict_per_epoch += acc/train_loader.batch_size #accross the epoch how many correct we have

      loss_per_epoch.append(loss.detach().numpy())
      metric_per_epoch.append(correct_predict_per_epoch/dataset_sizes)

      if epoch%100 ==0 : #plot info only every 100 epochs
        print(f'Epoch {epoch + 1}, Loss: {loss} Acc: {100*correct_predict_per_epoch/dataset_sizes}%')


      # backward + optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

    model.load_state_dict(best_model_wts)

    if plot == 'True' :
      plot_trainning_single(model.num_epochs, loss_per_epoch)

    return model

def train(full_model, dataloaders, dataset_sizes) :
  for i,model in enumerate(full_model.models):
    print('Pixel ', i)
    train_single_model(model,dataloaders, dataset_sizes)

#test functions 
def test_single_model(model, test_loader, test_dataset) :
  ''' give the prediction of a single pixel'''
  F1 = []
  accuracy = []
  pred = []

  for batch_x, batch_y in test_loader: #size = 1
      outputs = model(batch_x)
      pred.append( model.predict_label(outputs))

  f1, acc = metrics(batch_y, pred)

  return F1, acc


def test(trained_models, test_loader) :
  with torch.no_grad(): 
      F1= []
      accuracy = []
      for batch_x, batch_y in test_loader: #size = 1
          outputs = trained_models(batch_x)
          pred_pattern = trained_models.predict_pattern(outputs)
          f1, acc = metrics(batch_y,pred_pattern)

          #store
          F1.append(f1)
          accuracy.append(acc)

  return F1, accuracy

#validate functions 
def validate_train_single_model(trained_model, val_loader, plot= False):
   #init
   best_model_wts = copy.deepcopy(model.state_dict())
   acc_epoch = 0
   best_acc = 0
   criterion = torch.nn.SoftMarginLoss(reduction='mean')
   optimizer = torch.optim.SGD(trained_model.parameters(), lr= learning_rate, momentum=momentum)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  
   for epoch in range(num_epochs):
    for batch_x, batch_y in val_loader:
      trained_model.eval()

      outputs = trained_model(batch_x)

      preds = trained_model.predict_label(outputs)

      loss = criterion(outputs, batch_y) #mean loss per batch

      # Add regularization:  Full loss = data loss + regularization loss
      weight = trained_model.fc.weight.squeeze()

      if trained_model.reg_type == 'L1':  # add L1 (LASSO - Least Absolute Shrinkage and Selection Operator)
                                                      # loss which leads to sparsity.
          loss += trained_model.reg_term * torch.sum(torch.abs(weight))

      elif trained_model.reg_type == 'L2':   # add L2 (Ridge) loss
          loss += trained_model.reg_term * torch.sum(weight * weight)

      acc = torch.sum(preds == batch_y.data) #accross the batch how many correct we have
      acc_epoch += acc/val_loader.batch_size #accross the epoch how many correct we have

      if acc_epoch > best_acc:
          best_acc = acc_epoch
          best_model_wts = copy.deepcopy(trained_model.state_dict())

      if epoch%100 ==0 : #plot info only every 100 epochs
        print(f'Epoch {epoch + 1}, Loss: {loss} Acc: {100*acc_epoch}%')

   print('Best val Acc in percentage: {:.4f}'.format(best_acc*100.))

   # Load best model weights
   trained_model.load_state_dict(best_model_wts)
    

   if plot == 'True' :
      plot_trainning_single(best_model_wts.num_epochs, best_acc)
      
   return best_model_wts

#usefull functions
def plot_trainning_single(num_epoch, loss_per_epoch) :
  fig, ax = plt.subplots()

  x = np.linspace(0, num_epoch, num_epoch)
  y = np.linspace(0,1, num_epoch)

  ax.set_title('Loss along the trainning')
  ax.set_ylabel('Loss per epoch')
  ax.set_xlabel('Epochs')
  ax.plot(x,loss_per_epoch, '.', 'k')

def F1andscore(true, pred):
  TP = torch.sum((true == 1) & (pred == 1)) #true positive: if pred =1 both
  FP = torch.sum((true == -1) & (pred == 1)) #false positive: if pred = 1 and pat = -1
  FN = torch.sum((true == 1) & (pred == -1)) #False negative : if -1 in both
  TN = torch.sum((true == -1) & (pred == -1))

  # Calculate precision and recall
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)

  return 2 * (precision * recall) / (precision + recall), (TP + TN)
   
def metrics(true_patterns, pred_patterns):
  'return mean F1 over the batch, mean accuracy over the batch'
  F1 = []
  accuracy = []
  
  for pred, true in zip(pred_patterns, true_patterns):
    f1, acc = F1andscore(true, pred)
    F1.append(f1)
    accuracy.append(acc)  

  return np.mean(F1), np.mean(accuracy)
 
#saving functions
def save_model(model):
   """
   save : as a panda DataFrame
   - weight 
   - accuracy and F1 score
   - parameters
   - loss
   """
