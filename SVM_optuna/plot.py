import pandas as pd

from save_plot_functions import plot_testing, plot_training

"""
This module is a pipline to plot the outputs of the full-SVM training and testing phase. 
Plot are saved into Trials/Plots folder: 
   1. For the training phase the curves for:
   - Loss history per pixel
   - learning rate per pixel
   - accuracy per pixel
   - balanced accuracy per pixel 
   - F1 score per pixel

   2. For the testing phase the heat maps for:
   - F1 score per pixel 
   - accuracy per pixel
   - balanced accuracy per pixel 

"""

def main() : 
    # Path where are saved the results
    path_results = 'Trials'

    # Read the training and testing result from csv
    Training_results = pd.read_csv(path_results + '/training.csv')
    Testing_results = pd.read_csv(path_results + '/testing.csv')

    num_epochs = 300

    plot_training(Training_results, num_epochs, path_results)
    plot_testing(Testing_results, path_results)

if __name__ == "__main__":
    main()
