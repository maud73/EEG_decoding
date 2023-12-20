import pandas as pd
import ast

from save_plot_functions import plot_testing, plot_training

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
