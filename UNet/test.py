from model import UNet, UNet_dropout
from reproducibility import *
from Data_processing import *
from test_functions import get_predictions, visualize
from loss import FocalLoss
import numpy as np
import torch

"""
This module handles the pipeline of testing the UNet models with and without dropout, including visualization.

1. Load Data and Set Up Parameters:
   - `set_random_seeds()`: Ensures reproducibility by setting random seeds.
   - `get_data(file_path)`: EEG data loading and preprocessing.

2. Load Best Hyperparameters and Models:
   - Loads the best hyperparameters and models previously trained.

3. Get Predictions and Test Models:
   - Gets predictions and evaluates models on the test data: loss, accuracy, balanced accuracy, and F1 score for both models.

4. Visualizations:
   - Creates visualizations displaying the stimuli with the highest and lowest loss.

"""


def test():
    # Load data and set up parameters
    set_random_seeds()
    epochs, labels = get_data(file_path='/content/drive/MyDrive/Colab_Notebooks/ML/data/resampled_epochs_subj_0.pkl')
    batch_size = 64
    test_size = 0.2
    data_kwargs = dict(
    epochs=epochs,
    labels=labels,
    batch_size=batch_size,
    test_size=test_size,
    return_val_set=False
    )

    train_loader, test_loader = get_dataloaders(**data_kwargs)

    num_epochs_dropout = 18
    num_epochs_wo_dropout = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load best hyperparameters and models
    file_path = '/content/drive/MyDrive/Colab_Notebooks/best_hyperparams.pkl'
    with open(file_path, 'rb') as fp:
        best_hyp = pickle.load(fp)
    gamma = np.round(best_hyp['Param']['focus'], 4) # focal parameter

    filename_dropout = f'{num_epochs_dropout}_iters_{batch_size}batch_{gamma}gamma_dropout'
    dropout_unet_fp = f'/content/drive/MyDrive/Colab_Notebooks/{filename_dropout}.pth'
    dropout_unet = UNet_dropout(n_channels=1, n_classes=2, p_dropout=0.5).to(device)
    dropout_unet.load_state_dict(torch.load(dropout_unet_fp))
    dropout_unet = dropout_unet.double()
    dropout_unet.eval()

    filename_wo_dropout = f'{num_epochs_wo_dropout}_iters_{batch_size}batch_{gamma}gamma_wo_dropout'
    unet_fp = f'/content/drive/MyDrive/Colab_Notebooks/{filename_wo_dropout}.pth'
    unet = UNet(n_channels=1, n_classes=2).to(device)
    unet.load_state_dict(torch.load(unet_fp))
    unet = unet.double()
    unet.eval()


    # Get predictions and test models
    stimuli = [convert_from_id_to_grid(i) for i in range(1,61)]

    # Without dropout
    points_wo_dropout, losses_wo_dropout = get_predictions(stimuli, unet, device, test_loader, criterion=FocalLoss(gamma=gamma), num=len(test_loader.dataset))

    mean_loss = np.mean(np.array([point['loss'] for point in points_wo_dropout]))
    mean_accuracy = np.mean(np.array([point['accuracy'] for point in points_wo_dropout]))
    mean_soft_accuracy = np.mean(np.array([point['soft_accuracy'] for point in points_wo_dropout]))
    mean_f1 = np.mean(np.array([point['f1'] for point in points_wo_dropout]))

    print(f"Mean Loss: {mean_loss:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Balance Accuracy: {mean_soft_accuracy:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")

    # With dropout
    points_dropout, losses_dropout = get_predictions(stimuli, dropout_unet,device,test_loader,criterion=FocalLoss(gamma=gamma),num=len(test_loader.dataset))

    mean_loss = np.mean(np.array([point['loss'] for point in points_dropout]))
    mean_accuracy = np.mean(np.array([point['accuracy'] for point in points_dropout]))
    mean_soft_accuracy = np.mean(np.array([point['soft_accuracy'] for point in points_dropout]))
    mean_f1 = np.mean(np.array([point['f1'] for point in points_dropout]))

    print(f"Mean Loss: {mean_loss:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Balance Accuracy: {mean_soft_accuracy:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")

    # Visualization
    highest_loss_points = sorted(points_dropout, key=lambda x: x['loss'], reverse=True)[:5] # Get the 5 best and worst losses
    lowest_loss_points = sorted(points_dropout, key=lambda x: x['loss'])[:5]

    visualize(lowest_loss_points)
    visualize(highest_loss_points)

if __name__ == '__main__':
    test()
