
import torch
import mne
import pickle
import numpy as np

from torch.utils.data import Dataset, Subset
from torchvision import transforms

from mne.decoding import Scaler
from mne import create_info

from sklearn.model_selection import train_test_split    

from reproducibility import seed_worker


def load_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl'): 
    '''
    Loads the mne.EpochsArray data for one patient from a pickle file. This function is called in get_data().

    Args:
        file_path (string): The path to the pickle file. Default is an example file for Google Colab use
    
    Returns:
        epochs (mne.EpochsArray): All the EEG recordings for one patient
    '''
    with open(file_path, 'rb') as f:
        epochs = pickle.load(f)

    return epochs


def make_labels_grid_tensor(labels,convention_neg=False,two_channels=False):
    '''
    Converts the 60 numeric labels present in the mne events to a grid tensor matching the given stimuli. 
    This function is called in get_data().

    Args:
        labels (np.array): The labels of the stimuli from 1 to 60
        convention_neg (bool): Whether to use the convention of -1 for the negative class or not. Default is False
        two_channels (bool): Whether to return a 2-channel tensor or not, with the second channel being the inverse of the first one.
            Default is False, and is not used in the current implementation.

    Returns:
        labels_grid_tensor (torch.tensor): The labels in grid tensor format of shape (num_samples, 5, 5)

    '''
    labels_grid_tensor = torch.zeros((len(labels), 5, 5))
    for i in range(len(labels)):
        labels_grid_tensor[i] = torch.tensor(convert_from_id_to_grid(labels[i]))

    if two_channels:
        labels_grid_tensor_2_channels = torch.zeros((len(labels),2, 5, 5))
        labels_grid_tensor_2_channels[:,0,:,:] = labels_grid_tensor
        labels_grid_tensor_2_channels[:,1,:,:] = 1 - labels_grid_tensor
        labels_grid_tensor = labels_grid_tensor_2_channels

    if convention_neg:
        labels_grid_tensor[labels_grid_tensor == 0] = -1
    
    return labels_grid_tensor

def get_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0_corrected.pkl', convention_neg=False, two_channels=False):
    '''
    Loads the mne.EpochsArray data for one patient from a pickle file, and converts the labels to a grid tensor.
    This function is called in main().

    Args:
        file_path (string): The path to the pickle file. Default is an example file for Google Colab use
        convention_neg (bool): Whether to use the convention of -1 for the negative class or not. Default is False
        two_channels (bool): Whether to return a 2-channel tensor or not, with the second channel being the inverse of the first one.
            Default is False, and is not used in the current implementation.

    Returns:
        epochs (mne.EpochsArray): All the EEG recordings for one patient
        labels_grid_tensor (torch.tensor): The labels in grid tensor format of shape (num_samples, 5, 5)
    '''
    # Load data
    epochs = load_data(file_path)

    # Crop the data to match the time when the visual stimulus was on
    tmin = 0
    tmax = 0.746875
    epochs.crop(tmin=tmin, tmax=tmax)
    
    # Convert the labels to a grid tensor
    labels = epochs.events[:, 2] 
    labels = make_labels_grid_tensor(labels, convention_neg, two_channels)

    # Standardize data using mne library
    info = create_info(ch_names=epochs.ch_names, sfreq=epochs.info['sfreq'], ch_types='eeg') 
    scaler = Scaler(info=info, scalings='mean', with_mean=True, with_std=True)
    scaler.fit(epochs.get_data())
    epochs = scaler.transform(epochs.get_data())
    
    return epochs, labels

class EpochsDataset(Dataset):
    # Code adapted from https://github.com/mne-tools/mne-torch.git
    """Class to expose an MNE Epochs object as PyTorch dataset

    Parameters
    ----------
    epochs_data : 3d array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    epochs_labels : array of int, shape (n_epochs,)
        The epochs labels.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """
    def __init__(self, epochs_data, epochs_labels, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X)
        return X, y

def get_dataloaders(epochs, labels, batch_size, test_size, return_val_set=True,val_size=0.3):
    '''
    Splits the data into train, validation and test sets, and returns the corresponding dataloaders.
    This function is called in main(). Torch generators are defined to ensure reproducibility.

    Args:
        epochs (mne.EpochsArray): All the EEG recordings for one patient
        labels (torch.tensor): The labels in grid tensor format of shape (num_samples, 5, 5)
        batch_size (int): The batch size
        test_size (float): The proportion of the data to be used for testing
        return_val_set (bool): Whether to return a validation set or not. Default is True, but is set to false when using Optuna.
        val_size (float): The proportion of the data to be used for validation when return_val_set is True. Default is 0.3

    Returns:
        train_loader (torch.utils.data.DataLoader): The dataloader for the training set
        val_loader (torch.utils.data.DataLoader): The dataloader for the validation set
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set
    '''
    # Define the dataset class
    dataset_cls = EpochsDataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Split the data into different sets
    X_temp, epochs_test, y_temp, labels_test = train_test_split(epochs, labels, test_size=test_size, random_state=42)
    if return_val_set:
        epochs_train, epochs_val, labels_train, labels_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
        val_set = dataset_cls(
            epochs_data = epochs_val,
            epochs_labels = labels_val,
            transform=transform,)
        g_val = torch.Generator()
        g_val.manual_seed(42)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g_val)
        
    else:
        epochs_train, labels_train = X_temp, y_temp
       
    print(f'Dataset is split')

    # Define the dataloaders
    train_set = dataset_cls(
        epochs_data = epochs_train,
        epochs_labels = labels_train,
        transform=transform)
    g_train = torch.Generator()
    g_train.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g_train)

    test_set = dataset_cls(
        epochs_data = epochs_test,
        epochs_labels = labels_test,
        transform=transform)
    g_test=torch.Generator()
    g_test.manual_seed(42)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g_test)

    if return_val_set:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader

def get_valset(train_loader, val_size):
    '''
    Splits the training set into a pseudo-train and validation sets, and returns the validation set.
    This function is called in main() and used for Optuna validation.

    Args:
        train_loader (torch.utils.data.DataLoader): The dataloader for the training set
        val_size (float): The proportion of the data to be used for validation

    Returns:
        val_set (torch.utils.data.Subset): The validation set as a Subset of the training set
    '''
    trainset_size = len(train_loader.dataset)
    subset_size = val_size
    subset_indices = np.random.choice(trainset_size, size=int(subset_size * trainset_size), replace=False)
    val_set = Subset(train_loader.dataset,subset_indices)

    return val_set

def get_optuna_dataloaders(optuna_dataset, batch_size,optuna_val_size):
    '''
    Splits the optuna_dataset into a pseudo-train and validation sets, and returns the corresponding dataloaders.
    This function is called in main() and used for Optuna validation. Torch generators are defined to ensure reproducibility.

    Args:
        optuna_dataset (torch.utils.data.Dataset): The dataset to be split
        batch_size (int): The batch size
        optuna_val_size (float): The proportion of the data to be used for validation

    Returns:
        train_loader (torch.utils.data.DataLoader): The dataloader for the optuna-training set
        val_loader (torch.utils.data.DataLoader): The dataloader for the optuna-validation set
    '''
    # Extract indices from the train_loader dataset
    indices = list(range(len(optuna_dataset)))

    # Split indices into training and validation sets
    train_indices, val_indices = train_test_split(indices, test_size=optuna_val_size, random_state=42)

    # Create Subset datasets and DataLoaders for optuna training and validation
    train_set = Subset(optuna_dataset, train_indices)
    val_set = Subset(optuna_dataset, val_indices)
    g_train = torch.Generator()
    g_train.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g_train)
    g_val = torch.Generator()
    g_val.manual_seed(42)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g_val)
    
    return train_loader, val_loader

def convert_from_id_to_grid(id):
    '''
    Args:
        id: int, The id of the stimulus 
    Returns:
        grid: np.array, The corresponding grid of the stimulus
    '''
    grid = np.zeros((5, 5))
    if id == 1:
        grid[0, 0] = 1
        return grid 
    elif id == 2:
        grid[0, 1] = 1
        return grid
    elif id == 3:
        grid[0, 2] = 1
        return grid
    elif id == 4:
        grid[0, 3] = 1
        return grid
    elif id == 5:
        grid[0, 4] = 1
        return grid
    elif id == 6:
        grid[1, 0] = 1
        return grid
    elif id == 7:
        grid[1, 1] = 1
        return grid
    elif id == 8:
        grid[1, 2] = 1
        return grid
    elif id == 9:
        grid[1, 3] = 1
        return grid
    elif id == 10:
        grid[1, 4] = 1
        return grid
    elif id == 11:
        grid[2, 0] = 1
        return grid
    elif id == 12:
        grid[2, 1] = 1
        return grid
    elif id == 13:
        grid[2, 2] = 1
        return grid
    elif id == 14:
        grid[2, 3] = 1
        return grid
    elif id == 15:
        grid[2, 4] = 1
        return grid
    elif id == 16:
        grid[3, 0] = 1
        return grid
    elif id == 17:
        grid[3, 1] = 1
        return grid
    elif id == 18:
        grid[3, 2] = 1
        return grid
    elif id == 19:
        grid[3, 3] = 1
        return grid
    elif id == 20:
        grid[3, 4] = 1
        return grid
    elif id == 21:
        grid[4, 0] = 1
        return grid
    elif id == 22:
        grid[4, 1] = 1
        return grid
    elif id == 23:
        grid[4, 2] = 1
        return grid
    elif id == 24:
        grid[4, 3] = 1
        return grid
    elif id == 25:
        grid[4, 4] = 1
        return grid
    elif id == 26:
        grid[0:2, 0:2] = 1
        return grid 
    elif id == 27:
        grid[0:2, 1:3] = 1
        return grid
    elif id == 28:
        grid[0:2, 2:4] = 1
        return grid
    elif id == 29:
        grid[0:2, 3:5] = 1
        return grid
    elif id == 30:
        grid[1:3, 0:2] = 1
        return grid
    elif id == 31:
        grid[1:3, 1:3] = 1
        return grid
    elif id == 32:
        grid[1:3, 2:4] = 1
        return grid
    elif id == 33:
        grid[1:3, 3:5] = 1
        return grid
    elif id == 34:
        grid[2:4, 0:2] = 1
        return grid
    elif id == 35:
        grid[2:4, 1:3] = 1
        return grid
    elif id == 36:
        grid[2:4, 2:4] = 1
        return grid
    elif id == 37:
        grid[2:4, 3:5] = 1
        return grid
    elif id == 38:
        grid[3:5, 0:2] = 1
        return grid
    elif id == 39:
        grid[3:5, 1:3] = 1
        return grid
    elif id == 40:
        grid[3:5, 2:4] = 1
        return grid
    elif id == 41:
        grid[3:5, 3:5] = 1
        return grid
    elif id == 42:
        grid[0:3, 0:3] = 1
        return grid
    elif id == 43:
        grid[0:3, 2:5] = 1
        return grid
    elif id == 44:
        grid[1:4, 1:4] = 1
        return grid
    elif id == 45:
        grid[2:5, 0:3] = 1
        return grid
    elif id == 46:
        grid[2:5, 2:5] = 1
        return grid
    elif id == 47:
        grid[0:3, 0] = 1
        return grid
    elif id == 48:
        grid[0:3, 2] = 1
        return grid
    elif id == 49:
        grid[0:3, 4] = 1
        return grid
    elif id == 50:
        grid[2:5, 0] = 1
        return grid
    elif id == 51:
        grid[2:5, 2] = 1
        return grid
    elif id == 52:
        grid[2:5, 4] = 1
        return grid
    elif id == 53:
        grid[0, 0:3] = 1
        return grid
    elif id == 54:
        grid[2, 0:3] = 1
        return grid
    elif id == 55:
        grid[4, 0:3] = 1
        return grid
    elif id == 56:
        grid[0, 2:5] = 1
        return grid
    elif id == 57:
        grid[2, 2:5] = 1
        return grid
    elif id == 58:
        grid[4, 2:5] = 1
        return grid
    elif id == 59:
        grid[2, 0:5] = 1
        return grid
    elif id == 60:
        grid[0:5, 2] = 1
        return grid
