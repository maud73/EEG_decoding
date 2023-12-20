import torch


class SVM_pixel(torch.nn.Module):
  """
  SVM_pixel class, support vector machine model classify single pixel into the class 0 or 1.
  """
  def __init__(self,  input_size, num_classes=2):
    super(SVM_pixel, self).__init__()
    self.fc = torch.nn.Linear(input_size, num_classes)

    # To have an data type Float32 as an entry
    self.double()

  def forward(self, x):
    out = self.fc(x)
    return out

  def predict_label(self, outputs):
    _, preds = torch.max(outputs, 1)
    preds = preds.to(torch.int32)
    return preds

class SVM(torch.nn.Module):
  """
  SVM, support machine model for 5x5 image classification.
  """
  def __init__(self, device,  input_size, pixel_nb = 25):
    super(SVM, self).__init__()
    self.models = []

    for i in range(pixel_nb) :
      model = SVM_pixel(input_size) 
      model = model.to(device)
      self.models.append(model)

  def forward(self, x):
    for i,model in enumerate(self.models):
      out = model(x)
      if 'outs' in locals() :
        outs=torch.cat((outs,out), axis = 1)
      else : outs = out

    return outs

  def predict_pattern(self, outputs):
    for i in range(len(self.models)): 
      # Call the predict of the single SVM model with the right data pixel
      pred = self.models[i].predict_label(outputs[:,i,:])  

      if i ==0 : preds = pred
      else : preds = torch.cat((preds, pred), axis = 0)

    return preds.reshape([outputs.shape[0], outputs.shape[1]]) 

