import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
import os
import pandas as pd
import ast

def save_trial(df_training, df_testing, param, path_to_save):
  '''
  Save the results of the training and testing

  Args:
      df_training (pd.DataFrame): training results
      df_testing (pd.DataFrame): testing results
      param (dict): hyperparameters
      path_to_save (str): path to save the results
  '''
  # Change the parameters into DataFram with column labels
  df_param = pd.DataFrame.from_dict(param, orient='index')

  # Saving
  os.makedirs(path_to_save, exist_ok=True)

  df_param.to_csv(path_to_save + '/parameters.csv')
  df_testing.to_csv(path_to_save + '/testing.csv')
  df_training.to_csv(path_to_save + '/training.csv')

  print('trials saved in /trials directory')


def plot_training(Training_results, num_epochs, path_to_save):
  '''
  Plot the training results into 5x5 plots. Namely, the loss, the accuracy, the balanced accuracy, the learning rate 
  and the F1 score are repported.

  Args:
      Training_results (pd.DataFrame): training results
      num_epochs (int): number of epochs
      path_to_save (str): path to save the plots
  '''

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

  x = np.linspace(0,  num_epochs, num_epochs)

  
  for i, pixel in Training_results.iterrows() :
    # Finding the right index of the subplots
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
    
    training_loss = ast.literal_eval(str(pixel['Training loss']))
    training_acc = ast.literal_eval(str(pixel['Training accuracy']))
    training_wacc = ast.literal_eval(str(pixel['Training weighted accuracy']))
    training_lr = ast.literal_eval(str(pixel['Learning rate history']))
    training_f1 = ast.literal_eval(str(pixel['Training F1']))


    # Training loss
    axs1[n,m].plot(x, training_loss)

    # Training accuracy
    axs2[n,m].plot(x, training_acc) 

    # Training weighted accuracy
    axs3[n,m].plot(x, training_wacc)

    # Learning rate history
    axs4[n,m].plot(x, training_lr)

    # F1 score
    axs5[n,m].plot(x, training_f1)


  # Save plots into /Trials/plots folder 
  outpath = path_to_save + '/plots'

  os.makedirs(outpath, exist_ok=True)
  fig1.savefig(os.path.join(outpath,"Training_loss.png"))
  fig2.savefig(os.path.join(outpath,"Training_accuracy.png"))
  fig3.savefig(os.path.join(outpath, "Training_weighted_accuracy.png"))
  fig4.savefig(os.path.join(outpath, "Training_Learning_rate_history"))
  fig5.savefig(os.path.join(outpath, "Training_F1_score"))

  print('saving done in /trials/plots directory')

def plot_testing(Testing_results, path_to_save):
  '''
  Plot the training results into 5x5 plots. Namely, the accuracy, the balanced accuracy and the F1 score are repported.

  Args:
      Testing_results (pd.DataFrame): testing results
      path_to_save (str): path to save the plots
  '''
  Wacc = Testing_results['Testing singles weighted accuracy'].to_numpy(dtype = np.float64).reshape((5,5))
  f1 = Testing_results['Testing singles F1'].to_numpy(dtype = np.float64).reshape((5,5))
  acc = Testing_results['Testing singles accuracy'].to_numpy(dtype = np.float64).reshape((5,5))

  # Fig 1. Balanced accuracy
  fig1, ax1 = plt.subplots()
  im, cbar = heatmap(Wacc, ax=ax1, cmap="YlGn", cbarlabel="Balanced accuracy")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  # Fig 2. F1 score
  fig2, ax2 = plt.subplots()
  im, cbar = heatmap(f1, ax=ax2, cmap="YlGn", cbarlabel="F1 score")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  # Fig 3. Accuracy score
  fig3, ax3 = plt.subplots()
  im, cbar = heatmap(acc, ax=ax3, cmap="YlGn", cbarlabel="Accuracy")
  texts = annotate_heatmap(im, valfmt="{x:.4f}")
 
  # Save plot into Trials/plots directory 
  outpath = path_to_save +'/plots'
  os.makedirs(outpath, exist_ok=True)

  fig1.savefig(os.path.join(outpath,"Testing_wacc.png"))
  fig2.savefig(os.path.join(outpath,"Testing_f1.png"))
  fig3.savefig(os.path.join(outpath,"Testing_acc.png"))

  print('saving done in /trials/plots directory')

def save_prediction(true_patterns, pred_patterns, outpath, i):
  '''
  Save the True stimuli image vs. predicted stimuli. 

  Args:
      true_patterns (torch.Tensor): true stimuli
      pred_patterns (torch.Tensor): predicted stimuli
      outpath (str): path to save the plots
      i (int): pixel number
  '''
  # Saving path
  outpath = outpath + '/testing_pattern_example'
  os.makedirs(outpath, exist_ok=True)

  for true_pattern, pred_pattern in zip(true_patterns, pred_patterns) :
    
    # Plot the stimuli into a grid
    fig, _ = plot_pattern([true_pattern, pred_pattern])

    filname = f'Pred_vs_true_n{i}'

    # Saving
    fig.savefig(os.path.join(outpath, filname))

def plot_pattern(patterns):
  '''
  Plot the stimuli into a 5x5 grid

  Args:
      patterns (list): list of the stimuli to plot 

  Returns:
      fig (matplotlib.figure.Figure): figure
      ax (matplotlib.axes.Axes): axes
  '''
  fig, ax = plt.subplots(1, 2, figsize=(2*3,3))

  for i, pattern in enumerate(patterns) :
    Z = pattern.numpy().reshape([5,5])
    c = ax[i].pcolor(Z, cmap='binary')

  ax[0].set_title('True pattern')
  ax[1].set_title('Predict pattern')

  return fig, ax

def heatmap(data, cbarlabel, ax=None, cbar_kw=None, **kwargs):
    '''
    Create a heatmap from a numpy array and two lists of labels.

    Args:
      data (numpy array): data of shape (M, N)
      cbarlabel (string): title of the vertical bar
      ax (matplotlib.axes.Axes): instance to which the heatmap is plotted.  If not provided, use current axes or create a new one.  Optional.
      cbar_kw (dictionary):  with arguments to `matplotlib.Figure.colorbar`
      **kwargs: All other arguments are forwarded to `imshow`

    Returns:
      fig (matplotlib.figure.Figure): figure
      ax (matplotlib.axes.Axes): axes
    ''' 
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=21)

    # No ticks 
    ax.set_yticks(np.arange(len(data)), labels=[])
    ax.set_xticks(np.arange(len(data)), labels=[])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False,labeltop=False, labelbottom=False)

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

def annotate_heatmap(im, valfmt="{x:.2f}", **textkw):
    '''
    A function to annotate a heatmap

    Args:
      im (AxesImage): to be labeled
      data Data used to annotate.  If None, the image's data is used.  Optional.
      valfmt (string or matplotlib.ticker.Formatter): The format of the annotations inside the heatmap. If string shoudl be
        string format method, e.g. "$ {x:.2f}"
      **kwargs: All other arguments are forwarded to each call to `text` used to creat the text labels
    '''

    # Use the image data to label
    data = im.get_array()

    # Normalize the threshold to the images color range
    threshold = im.norm(data.max())/2.
    
    # Set default alignment to center
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Set the text color to balck and white
    textcolors=("black", "white")

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

