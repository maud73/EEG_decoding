# To use in Google Colab : see in README.md

import torch
import mne
import pickle
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from mne.decoding import Scaler
from mne import create_info

# Load the MNE object from the .pkl file
def load_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl'): 
    with open(file_path, 'rb') as f:
        epochs = pickle.load(f)

    return epochs

# Code from https://github.com/mne-tools/mne-torch.git

def get_data(file_path='/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl'):
    # Load data
    epochs = load_data(file_path)
    # Crop the data to keep it only when the visual stimulus was on
    tmin = 0
    tmax = 0.746875
    epochs.crop(tmin=tmin, tmax=tmax)
    labels = epochs.events[:, 2] 
    # Normalize data using mne library
    info = create_info(ch_names=epochs.ch_names, sfreq=epochs.info['sfreq'], ch_types='eeg') 
    scaler = Scaler(info=info, scalings=None, with_mean=True, with_std=True)
    scaler.fit(epochs.get_data())
    epochs = scaler.transform(epochs.get_data())
    
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
from sklearn.model_selection import train_test_split    
def get_dataloaders(
    epochs,
    labels,
    batch_size):
    dataset_cls = EpochsDataset

    transform = transforms.Compose(
        [transforms.ToTensor() ]
    )

    epochs_train, epochs_test, labels_train, labels_test = train_test_split(epochs, labels, test_size=0.3, random_state=42)
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
        #pin_memory=torch.cuda.is_available(),
        #drop_last=False,
        #num_workers=2,
    )

    val_set = dataset_cls(
        epochs_data = epochs_test,
        epochs_labels = labels_test,
        transform=transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
