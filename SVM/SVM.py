from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import os

# ================ MODELS ================

class SVM_pixel(torch.nn.Module) :
  """
  SVM_pixel class, support vector machine model classify single pixel
  into the class -1 or 1.
  """
  def __init__(self,  input_size, num_classes=2):
    super(SVM_pixel, self).__init__()
    self.fc = torch.nn.Linear(input_size, num_classes)

    #to have an data type Float32 as an entry
    self.double()

  def forward(self, x):
    out = self.fc(x)
    return out

  def predict_label(self, outputs):
    _, preds = torch.max(outputs, 1)
    mask = [preds==0]
    preds[mask]= -1
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
      model = SVM_pixel(input_size) #inputsize = 192*128 = nb of feature
      model = model.to(device)
      self.models.append(model)

  def forward(self, x):
    for i,model in enumerate(self.models):
      out = model(x)
      if 'outs' in locals() :
        outs=torch.cat((outs,out), axis = 1)
        # outs=torch.concatenate((outs,out), axis = 1)
      else : outs = out

    return outs

  def predict_pattern(self, outputs):
    for i in range(len(self.models)): #shape outputs = [50, 25, 2]) batch_size, num_pixel, nb_class
      pred = self.models[i].predict_label(outputs[:,i,:])  #take only the right pixel

      if i ==0 : preds = pred
      else : preds = torch.cat((preds, pred), axis = 0)

    return preds.reshape([outputs.shape[0], outputs.shape[1]]) #size batch_size, nb of pixel

# ================ TRAINING FUNCTIONS ================

def train_single_model(model,
                       train_loader,
                       num_pixel,
                       num_epoch,
                       device,
                       weight_,
                       param):

    #crit, optim and scheduler
    criterion =  torch.nn.HingeEmbeddingLoss(param['loss_margin']) #MultiLabelSoftMarginLoss(weight = weight_) 

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['lr'],
                                 betas=(param['beta1'], param['beta2']),
                                 eps=1e-08,
                                 weight_decay=param['weight_decay'])

    scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=param['gamma'])

    #regularization parameters
    reg_type = param['reg_type']
    reg_term = param['reg_term']


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

      model.train()  # Set model to training model

      for batch_x, batch_y in train_loader:
        batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)

        #running on gpu
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        preds = model.predict_label(outputs)
        loss = criterion(outputs, batch_y)

        # Add regularization:  Full loss = data loss + regularization loss
        weight = model.fc.weight.squeeze()

        if reg_type == 'L1':  # add L1 (LASSO - Least Absolute Shrinkage and Selection Operator)
                                                        # loss which leads to sparsity.
            loss += reg_term * torch.sum(torch.abs(weight))

        elif reg_type == 'L2':   # add L2 (Ridge) loss
            loss += reg_term * torch.sum(weight * weight)
        
        '''
        if torch.isnan(torch.any(weight)) : 
           print('NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')

        '''

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
          
        running_loss.append(loss.cpu().item())
        running_corrects += torch.mean((preds == batch_y.data).float()).cpu()
        
        wacc = balanced_accuracy_score(batch_y.view(-1).cpu(),preds.view(-1).cpu()) # Mean wacc on the batch
        running_wacc += wacc
        f1 = f1_score(batch_y.view(-1).cpu(),preds.view(-1).cpu()) # Mean f1 on the batch
        running_f1 += f1

      #save the loss, accuracy, lr, wacc and f1 for each epoch
      lr_history.append(scheduler.get_last_lr()[0])
      running_loss = torch.nanmean(torch.tensor(running_loss))
      loss_per_epoch.append(running_loss)
      acc_per_epoch.append(running_corrects / len(train_loader))
      wacc_per_epoch.append(running_wacc / len(train_loader))
      f1_per_epoch.append(running_f1 / len(train_loader))


      if epoch%100 ==0 : #plot info only every 100 epochs
        print(f'Epoch: {epoch}',
              'Loss: {:.4f}'.format(running_loss),
              'Acc: {:.4f}'.format(running_corrects/ len(train_loader)),
              'weighted Acc: {:.4f}'.format(running_wacc / len(train_loader)),
              'F1: {:.4f}'.format(running_f1 / len(train_loader)))

    return loss_per_epoch, acc_per_epoch, wacc_per_epoch, f1_per_epoch, lr_history

def train(full_model,
          train_loader,
          device,
          weights_,
          num_epoch,
          hyperparam
          ) :

  to_store = pd.DataFrame(columns=['Pixel n°',
                                   'Training loss',
                                   'Learning rate history',
                                   'Training accuracy',
                                   'Training weighted accuracy',
                                   'Training F1'])

  for i, model in enumerate(full_model.models):
    print('Pixel ', i )
    weight = weights_[i].to(device)

    loss_per_epoch, acc_per_epoch, wacc_per_epoch, f1_per_epoch, lr_history = train_single_model(model,
                                                                                   train_loader,
                                                                                   i,
                                                                                   num_epoch,
                                                                                   device, 
                                                                                   weight, 
                                                                                   hyperparam.iloc[i])

    #Store
    to_store.loc[len(to_store.index)] =  [i, loss_per_epoch,lr_history,acc_per_epoch, wacc_per_epoch, f1_per_epoch]

  #save the model
  torch.save(full_model, 'SVMmodel.pth')

  print('the model is saved under SVMmodel.pth!')

  return to_store

# ================ TEST FUNCTIONS ================

#test functions
def test_single_model(model, test_loader, num_pixel, weight, device) :
  ''' give the prediction of a single pixel'''
  #running_corrects =0
  running_wacc=0
  running_f1 =0
  with torch.no_grad():
    for batch_x, batch_y in test_loader:

        batch_x, batch_y = resize_batch(batch_x, batch_y, num_pixel)

        #running on gpu
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        preds = model.predict_label(outputs)

        #running_corrects += torch.mean((preds == batch_y.data).float()).cpu()
        wacc = accW(batch_y.cpu(), preds.cpu(), weight.cpu())
        running_wacc += wacc

        f1 = f1_score(batch_y.cpu(), preds.cpu())
        running_f1 += f1

    #acc = running_corrects / len(test_loader.dataset)
    Wacc = running_wacc / len(test_loader.dataset)
    F1 = running_f1/len(test_loader.dataset)

  return F1, Wacc

def test(trained_model, test_loader, outpath,weights, device) : 

  to_store = pd.DataFrame(columns=['Testing singles weighted accuracy', 'Testing singles accuracy'])

  #test 1 : the single test over each pixel
  wacc_single = []
  f1_single = []

  for i, model in enumerate(trained_model.models) :
    f1, wacc = test_single_model(model, test_loader, i, weights[i], device)
    wacc_single.append(wacc)
    f1_single.append(f1)

  to_store['Testing singles weighted accuracy'] = wacc_single
  to_store['Testing singles F1'] = f1_single
  i = 0
  with torch.no_grad():
  
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.flatten(2)
        batch_y = batch_y.flatten(1)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = trained_model(batch_x)
        i+=1
        if i%100 ==0 :
          pred_pattern = trained_model.predict_pattern(outputs)
          save_prediction(batch_y.cpu(), pred_pattern.cpu(), outpath, i) 

  print('saving example pattern into trials/testing_patterns_example')
  return to_store

# ================ ADDITIONNAL FUNCTIONS ================
#resize function for the batches
def resize_batch(batch_x, batch_y, num_pixel):
  #reshape x_batch
  batch_x= batch_x.flatten(2)
  shape0 = batch_x.shape[0]
  shape1 = batch_x.shape[2]
  batch_x = batch_x.reshape([shape0, shape1])

  #take the right batch for the right pixel
  batch_y = batch_y.flatten(1)

  #take the right pixel
  batch_y = batch_y[:,num_pixel]
  batch_y = batch_y.reshape(batch_y.shape[0], 1)

  #y label should be int
  batch_y= batch_y.to(torch.int32)

  return batch_x, batch_y

def balance_weight(labels) :
  #finding the right weight
  labs = labels.flatten(1)

  alphas = []
  for i in range (labs.shape[1]) :
    lab = labs[:,i]
    item, count = torch.unique(lab, return_counts=True)
    alpha = count/lab.shape[0]
    alphas.append(1-alpha)

  return item, alphas

#metric
def accW(true, pred, weight):
  w=[]
  for y in true :
    if y == -1 : w.append(weight[0]) #largerly more -1 -> more accurate if we manage to predict a 1
    else : w.append(weight[1])

  acc = balanced_accuracy_score(true.flatten(), pred, sample_weight=w)

  return acc

#saving functions
def save_trial(df_training, df_testing, param, path_to_save):

  #add the parameters column
  df_param = pd.DataFrame.from_dict(param, orient='index')

  path = path_to_save

  import os
  os.makedirs(path, exist_ok=True)

  df_param.to_csv(path + '/parameters.csv')
  df_testing.to_csv(path + '/testing.csv')
  df_training.to_csv(path + '/training.csv')

  print('trials saved in /trials directory')

#plotting and save functions
def plot_training(Training_results, num_epochs, path_to_save) :
  #columns=['Pixel n°', 'Training loss','Learning rate history', 'Training accuracy','Training weighted accuracy']

  #plot the 25 training loss and accuracy over the epochs in a subplot + one point for the best accuracy
  fig1, axs1 = plt.subplots(5, 5, figsize=(30,30))
  fig1.suptitle('Training loss', fontsize=40)
  fig2, axs2 = plt.subplots(5, 5, figsize=(30,30))
  fig2.suptitle('Training accuracy', fontsize=40)
  fig3, axs3 = plt.subplots(5, 5, figsize=(30,30))
  fig3.suptitle('Training weighted accuracy', fontsize=40)
  fig4, axs4 = plt.subplots(5, 5, figsize=(30,30))
  fig4.suptitle('Learning rate history', fontsize=40)
  fig5, axs5 = plt.subplots(5, 5, figsize=(30,30))
  fig5.suptitle('Training F1 score', fontsize=40)

  #x axis are the nb of epochs
  x = np.linspace(0,  num_epochs, num_epochs)

  for i, pixel in Training_results.iterrows() :
    n = i%5
    if i < 5:
      m = 0
    if i >= 5 and i < 10:
      m = 1
    if i >=10 and i < 15:
      m=2
    if i >= 15 and i < 20 :
      m = 3
    if i >=20 and i < 25:
      m = 4

    #'Training_loss'
    axs1[n,m].plot(x, pixel[1])

    # 'Training_accuracy'
    axs2[n,m].plot(x, pixel[3])

    # 'Training weighted accuracy'
    axs3[n,m].plot(x, pixel[4])

    #'Learning rate history'
    axs4[n,m].plot(x, pixel[2])

    #'F1 score'
    axs5[n,m].plot(x, pixel[5])


  #save plot into trials directory (same as the csv files)
  outpath = path_to_save + '/plots'

  import os
  os.makedirs(outpath, exist_ok=True)

  fig1.savefig(os.path.join(outpath,"Training_loss.png"))
  fig2.savefig(os.path.join(outpath,"Training_accuracy.png"))
  fig3.savefig(os.path.join(outpath, "Training_weighted_accuracy.png"))
  fig4.savefig(os.path.join(outpath, "Training_Learning_rate_history"))
  fig5.savefig(os.path.join(outpath, "Training_F1_score"))

  print('saving done in /trials/plots directory')

def plot_testing(Testing_results, path_to_save):
  #columns= ['Testing singles weighted accuracy', 'Testing singles accuracy']
  #heat map 5x5 for Testing singles F1

  Wacc = Testing_results['Testing singles weighted accuracy'].to_numpy(dtype = np.float64).reshape((5,5))
  f1 = Testing_results['Testing singles F1'].to_numpy(dtype = np.float64).reshape((5,5))
  acc = Testing_results['Testing singles accuracy'].to_numpy(dtype = np.float64).reshape((5,5))

  fig1, ax1 = plt.subplots()
  im, cbar = heatmap(Wacc, [], [], ax=ax1,
                   cmap="YlGn", cbarlabel="Balanced accuracy")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  fig2, ax2 = plt.subplots()
  im, cbar = heatmap(f1, [], [], ax=ax2,
                   cmap="YlGn", cbarlabel="F1 score")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  fig3, ax3 = plt.subplots()
  im, cbar = heatmap(acc, [], [], ax=ax3,
                   cmap="YlGn", cbarlabel="Accuracy")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")
 
  #save plot into trials directory (same as the csv files)
  outpath = path_to_save +'/plots'
  import os
  os.makedirs(outpath, exist_ok=True)

  fig1.savefig(os.path.join(outpath,"Testing_wacc.png"))
  fig2.savefig(os.path.join(outpath,"Testing_f1.png"))
  fig3.savefig(os.path.join(outpath,"Testing_acc.png"))

  print('saving done in /trials/plots directory')

def save_prediction(true_patterns, pred_patterns, outpath, i) :
  import os

  outpath = outpath + '/testing_pattern_example'
  os.makedirs(outpath, exist_ok=True)
  for true_pattern, pred_pattern in zip(true_patterns, pred_patterns) :

    fig, ax = plot_pattern([true_pattern, pred_pattern])
    filname = f'Pred_vs_true_n{i}'
    fig.savefig(os.path.join(outpath, filname))

def plot_pattern(patterns) :
  from matplotlib.colors import LogNorm
  fig, ax = plt.subplots(1, 2, figsize=(2*3,3))

  for i, pattern in enumerate(patterns) :
    Z = pattern.numpy().reshape([5,5])
    c = ax[i].pcolor(Z, cmap='binary')

  ax[0].set_title('true pattern')
  ax[1].set_title('predict pattern')

  return fig, ax

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=21)

    # no ticks 
    ax.set_yticks(np.arange(len(data)), labels=[])
    ax.set_xticks(np.arange(len(data)), labels=[])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False,
                   labeltop=False, labelbottom=False)
    

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    from matplotlib.ticker import StrMethodFormatter
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

#======= OPTUNA HYPERPARAMTER TUNNING ======
def find_hyperparam(path_to_save,device, weight_loss,input_size,o_train_loader, o_val_loader, num_epochs, n_trials, num_pixels = 25):
  '''return a pd.serie hyperparam[num_pixel][hyperparameter] '''

  optuna_result = pd.DataFrame(columns = ["Number of finished trials",
                                          "Number of pruned trials",
                                          "Number of complete trials",
                                          "Best accuracy",
                                          "Param"])

  to_return = pd.DataFrame(columns = ['lr', 'beta1', 'beta2', 'weight_decay', 'reg_term', 'reg_type', 'step_size', 'gamma', 'loss_margin'])
  for i in range(num_pixels):
    print('Pixel n°',i)
    weight = weight_loss[i].to(device)
    df = run_optuna(i, weight, device, input_size, o_train_loader, o_val_loader, num_epochs, n_trials)


    optuna_result.loc[len(optuna_result.index)] =  [df["Number of finished trials"],
                                                    df["Number of pruned trials"],
                                                    df["Number of complete trials"],
                                                    df["Best accuracy"],
                                                    df["Param"]]

    to_return.loc[len(to_return.index)] = [df["Param"]['lr'],
                                           df["Param"]["beta1"],
                                           df["Param"]["beta2"],
                                           df["Param"]['weight_decay'],
                                           df["Param"]['reg_term'],
                                           df["Param"]['reg_type'],
                                           df["Param"]['step_size'],
                                           df["Param"]['gamma'],
                                           df['Param']['loss_margin']]

  path = path_to_save

  os.makedirs(path, exist_ok=True)

  optuna_result.to_csv(path + '/optuna_running.csv')

  to_return.to_csv(path + '/hyperparam.csv')

  print('hyperparameters saved in /trials directory')

  return to_return

def run_optuna(num_pixel, weight_, device, input_size, o_train_loader, o_val_loader, num_epochs, n_trials):
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

    #optimizer hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 1, log=False)
    beta2 = trial.suggest_float("beta2", 0.9, 1, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    #regularization hyperparameters
    reg_term = trial.suggest_float('reg_term', 1e-5, 1e-1, log=True)
    reg_type = trial.suggest_categorical("reg_type", ["L1", "L2"])

    #scheduler hyperparameters
    step_size = trial.suggest_float('step_size', 5, 10)
    gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)

    #loss
    loss_margin = trial.suggest_float('loss_margin', 0, 5)

    model = SVM_pixel(input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=weight_decay)
    #criterion = torch.nn.MultiLabelSoftMarginLoss(weight = weight_)
    criterion = torch.nn.HingeEmbeddingLoss(margin = loss_margin)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training of the model.
    for epoch in range(num_epochs):
      model.train()
      for batch_x, batch_y in o_train_loader:
        batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        output = model(batch_x)
        loss = criterion(output, batch_y)

        # Add regularization:  Full loss = data loss + regularization loss
        weight = model.fc.weight.squeeze()
        if reg_type == 'L1':  # add L1 (LASSO - Least Absolute Shrinkage and Selection Operator)
                                                        # loss which leads to sparsity.
          loss += reg_term * torch.sum(torch.abs(weight))

        elif reg_type == 'L2':   # add L2 (Ridge) loss
          loss += reg_term * torch.sum(weight * weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

      # Validation of the model.
      model.eval()
      ACC = 0
      #F1 = 0
      with torch.no_grad():
        for batch_x, batch_y in  o_val_loader:
          batch_x, batch_y = resize_batch(batch_x, batch_y,  num_pixel)
          batch_x, batch_y = batch_x.to(device), batch_y.to(device)
          output = model(batch_x)
          pred = model.predict_label(output)

          acc = balanced_accuracy_score(batch_y.view(-1).cpu(),pred.view(-1).cpu())
          ACC += acc

          #f1 = f1_score(batch_y.view(-1).cpu(),pred.view(-1).cpu())
          #F1 += f1

      ACC_= ACC / len(o_val_loader.dataset)
      #F1 /= len(o_val_loader.dataset) 
      trial.report(ACC_, epoch)

      # Handle pruning based on the intermediate value.
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return ACC_

