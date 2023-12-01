# To use in Google Colab : see in README.md

import torch
import mne
import pickle
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from mne.decoding import Scaler
from mne import create_info

from sklearn.model_selection import train_test_split    


# Load the MNE object from the .pkl file
def load_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl'): 
    with open(file_path, 'rb') as f:
        epochs = pickle.load(f)

    return epochs

# Code from https://github.com/mne-tools/mne-torch.git

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

def make_labels_grid_tensor(labels, get_alpha=False,convention_neg=False,two_channels=False):
    '''
    Args:
        labels: np.array, The labels of the stimuli
        convention_neg: bool, If True, the 0s are replaced by -1s
        two_channels: bool, If True, the tensor is duplicated to have two channels that are the opposite of each other
    Returns:
        labels_grid_tensor: torch.tensor, The tensor of the labels in grid form
    '''
    labels_grid_tensor = torch.zeros((len(labels), 5, 5))
    for i in range(len(labels)):
        labels_grid_tensor[i] = torch.tensor(convert_from_id_to_grid(labels[i]))

    if get_alpha:
        alpha = make_alpha(labels_grid_tensor)

    if two_channels:
        labels_grid_tensor_2_channels = torch.zeros((len(labels),2, 5, 5))
        labels_grid_tensor_2_channels[:,0,:,:] = labels_grid_tensor
        labels_grid_tensor_2_channels[:,1,:,:] = 1 - labels_grid_tensor
        labels_grid_tensor = labels_grid_tensor_2_channels

    if convention_neg:
        labels_grid_tensor[labels_grid_tensor == 0] = -1
    
    if get_alpha:
        return labels_grid_tensor, alpha

    else:
        return labels_grid_tensor
    
def make_alpha(labels_grid_tensor):
    alpha = np.mean(labels_grid_tensor, axis=0)
    return alpha

def get_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl', get_alpha=False, convention_neg=False, two_channels=False):
    # Load data
    epochs = load_data(file_path)
    # Crop the data to keep it only when the visual stimulus was on
    tmin = 0
    tmax = 0.746875
    epochs.crop(tmin=tmin, tmax=tmax)
    labels = epochs.events[:, 2] 
    # Convert the labels to a grid tensor
    if get_alpha:
        labels, alpha = make_labels_grid_tensor(labels,get_alpha, convention_neg, two_channels)
    else:
        labels = make_labels_grid_tensor(labels,get_alpha, convention_neg, two_channels)
    # Normalize data using mne library
    info = create_info(ch_names=epochs.ch_names, sfreq=epochs.info['sfreq'], ch_types='eeg') 
    scaler = Scaler(info=info, scalings=None, with_mean=True, with_std=True)
    scaler.fit(epochs.get_data())
    epochs = scaler.transform(epochs.get_data())
    
    if get_alpha:
        return epochs, labels, alpha
    else:
        return epochs, labels

class EpochsDataset(Dataset):
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


def get_dataloaders(
    epochs,
    labels,
    batch_size):
    dataset_cls = EpochsDataset

    transform = transforms.Compose(
        [transforms.ToTensor() ]
    )

    # Assuming 'X' is your feature set and 'y' is your target variable
    X_temp, epochs_test, y_temp, labels_test = train_test_split(epochs, labels, test_size=0.3, random_state=42)
    epochs_train, epochs_val, labels_train, labels_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
    print(f'Dataset is split')

    train_set = dataset_cls(
        epochs_data = epochs_train,
        epochs_labels = labels_train,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the iteration order over the dataset
        pin_memory=torch.cuda.is_available(),
        #drop_last=False,
        #num_workers=2,
    )


    val_set = dataset_cls(
        epochs_data = epochs_val,
        epochs_labels = labels_val,
        transform=transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


    test_set = dataset_cls(
        epochs_data = epochs_test,
        epochs_labels = labels_test,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
