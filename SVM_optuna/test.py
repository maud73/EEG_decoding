from test_functions import test
from model import SVM
from Data_processing import get_data, get_dataloaders
from helpers import balance_weight, resize_batch
from reproducibility import *
from save_plot_functions import plot_testing

def main():
    # === Data Loading ===
    set_random_seeds()
  
    # Parameters 
    test_size = 0.2
    batch_size = 64

    # Path 
    file_path = 'resampled_epochs_subj_0.pkl'
    path_to_save = 'Trials'

    epochs, labels = get_data(file_path, convention_neg=False)
    _, test_loader = get_dataloaders(epochs[:155], labels[:155], batch_size, test_size, return_val_set=False) #for debugs
  
    # Find the label ratio
    _ , weights = balance_weight(labels)

    # Size of the input datapoint
    input_size = test_loader.dataset[:][0].shape[2]*test_loader.dataset[:][0].shape[0]

    # Defining the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model 
    filename_model = 'SVMmodel.pth'
    my_SVM = torch.load(filename_model)
    my_SVM = my_SVM.to(device)

    # Set to eval mode
    my_SVM.eval()

    # Testing on the testing loader
    Testing_results = test(my_SVM, test_loader, path_to_save, device) 

    plot_testing(Testing_results, path_to_save)

if __name__ == "__main__":
    main()

