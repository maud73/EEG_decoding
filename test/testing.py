import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


def main() : 
    df_testing = pd.read_csv('testing.csv')
    plot_testing(df_testing, 'plots')

if main == '__main__':
   main()


def plot_testing(Testing_results, path_to_save):
  #columns= ['Testing singles weighted accuracy', 'Testing singles accuracy']
  #heat map 5x5 for Testing singles F1

  Wacc = Testing_results['Testing singles weighted accuracy'].to_numpy(dtype = np.float64).reshape((5,5))
  Acc = Testing_results['Testing singles accuracy'].to_numpy(dtype = np.float64).reshape((5,5))

  fig1, ax1 = plt.subplots()

  im, cbar = heatmap(Wacc, ['0', '1', '2', '3', '4'], ['0', '1', '2', '3', '4'], ax=ax1,
                   cmap="YlGn", cbarlabel="F1 per pixel range ")

  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  fig1.suptitle('Testing weighted accuracy', fontsize=16)

  fig2, ax2 = plt.subplots()

  im, cbar = heatmap(Acc, ['0', '1', '2', '3', '4'], ['0', '1', '2', '3', '4'], ax=ax2,
                   cmap="YlGn", cbarlabel="accuracy per pixel range")

  texts = annotate_heatmap(im, valfmt="{x:.4f}")

  fig2.suptitle('Testing accuracy score', fontsize=16)

  #save plot into trials directory (same as the csv files)
  outpath = path_to_save +'/plots'
  import os
  os.makedirs(outpath, exist_ok=True)

  fig1.savefig(os.path.join(outpath,"Testing_wacc.png"))
  fig2.savefig(os.path.join(outpath,"Testing_acc.png"))

  print('saving done in /trials/plots directory')

def save_prediction(true_patterns, pred_patterns, outpath) :
  import os

  outpath = outpath + '/testing_pattern_example'
  os.makedirs(outpath, exist_ok=True)
  i = 0
  for true_pattern, pred_pattern in zip(true_patterns, pred_patterns) :

    fig, ax = plot_pattern([true_pattern, pred_pattern])
    i+= 1
    filname = f'Pred_vs_true_n{i}'
    fig.savefig(os.path.join(outpath, filname))

  print('saving pattern into trials/testing_patterns_example')

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
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

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