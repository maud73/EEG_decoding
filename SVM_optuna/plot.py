import pandas as pd

from save_plot_fonctions import plot_testing, plot_training
from main import num_epochs

def main() : 
    # Path where are saved the results
    path_results = '/Trials'

    # Read the training and testing result from csv
    Training_results = pd.read_csv(path_results + '/training.csv')
    Testing_results = pd.read_csv(path_results + '/testing.csv')

    plot_training(Training_results, num_epochs, path_results)
    plot_testing(Testing_results, path_results)

if __name__ == "__main__":
    main()
