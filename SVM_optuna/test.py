from test_functions import test
from model import SVM
from Data_processing import get_data, get_dataloaders
from helpers import balance_weight
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
    SVMmodel = SVM(device, input_size).to(device)

    SVMmodel.load_state_dict(torch.load(filename_model))
    SVMmodel = SVMmodel.double()
    SVMmodel.eval()

    # Testing on the testing loader
    Testing_results = test(SVMmodel, test_loader, path_to_save, device) 

    plot_testing(Testing_results, path_to_save)

if __name__ == "__main__":
    main()

