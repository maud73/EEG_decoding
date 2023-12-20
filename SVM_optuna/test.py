from test_functions import test
from model import SVM
from Data_processing import get_data, get_dataloaders, convert_from_id_to_grid
from reproducibility import *
from save_plot_functions import save_test, save_prediction

"""
This module pipeline aims to test an existing full-SVM model, wich corresponds to 25 SVM model.

1. Data Loading:
   - `set_random_seeds()`: Ensures reproducibility by setting random seeds.
   - `get_data(file_path)`: EEG data loading and preprocessing.

2. Model Testing:
   - Trains the full SVM model.

3. Saving
   - Save the outputs of the test: Accuracy, balanced accuracy and F1 score of the full test set.
   - Save the mean, 
"""

def main():
    # === Data Loading ===
    set_random_seeds()
  
    # Parameters 
    test_size = 0.2
    batch_size = 64

    # Path 
    file_path = 'data/resampled_epochs_subj_0.pkl'
    path_to_save = 'Trials'

    epochs, labels = get_data(file_path, convention_neg=False)
    _, test_loader = get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=False) #for debugs
  
    # Defining the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Model =====
    # Load the model 
    filename_model = 'SVMmodel.pth'
    my_SVM = torch.load(filename_model)
    my_SVM = my_SVM.to(device)

    # Get predictions and test models
    stimuli = [convert_from_id_to_grid(i) for i in range(1,61)]

    # Set to eval mode
    my_SVM.eval()

    # ===== Testing =====
    # Testing on the testing loader
    Testing_results, points = test(my_SVM, test_loader, path_to_save, device, stimuli) 

    # Mean metrics:
    mean_accuracy = np.mean(np.array([point['accuracy'] for point in points]))
    mean_soft_accuracy = np.mean(np.array([point['soft_accuracy'] for point in points]))
    mean_f1 = np.mean(np.array([point['f1'] for point in points]))

    # ===== Saving =====
    save_test(Testing_results,[mean_accuracy,mean_soft_accuracy, mean_f1], path_to_save)

    highest_points = sorted(points, key=lambda x: x['accuracy'], reverse=True)[:5] # Get the 5 best and worst losses
    lowest_loss_points = sorted(points, key=lambda x: x['accuracy'])[:5]

    # Save the Best ans Worst predictions
    save_prediction(highest_points, path_to_save, ind = 'hight')
    save_prediction(lowest_loss_points, path_to_save, ind= 'low')

if __name__ == "__main__":
    main()



