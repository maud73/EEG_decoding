from Data_processing import get_dataloaders, get_data
from train_functions import run_training
from reproducibility import set_random_seeds

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import UNet, UNet_dropout
from loss import FocalLoss
from test import test
import os 
import pickle


def main():
        print("UNet_without_dropout")
        # === Set random seeds for reproducibility ===
        set_random_seeds()
        # === Load the data ===        
        file_path = 'data/resampled_epochs_subj_0.pkl'
        epochs, labels = get_data(file_path)
        test_size = 0.2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For the dropout UNet
        #num_epochs = 18
        # For the classical UNet
        num_epochs = 8
        
        batch_size = 64 
        data_kwargs = dict(
        epochs=epochs,
        labels=labels, 
        batch_size=batch_size,
        test_size=test_size, 
        return_val_set=False
        )
        
        train_loader, test_loader = get_dataloaders(**data_kwargs)
        print("Data loaded")

        # === Define the UNet and the training hyperparameters ===
        #p_dropout = 0.5
        #model = UNet_dropout(n_channels=1, n_classes=2, p_dropout=p_dropout).to(device)
        model = UNet(n_channels=1, n_classes=2).to(device)
        model = model.double()
        optimizer_kwargs = dict(
        lr=3.998e-5,
        weight_decay=0.0002567,
        betas = (0.9520, 0.9986),
        )
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
        eta_min=2.600e-7,
        )
        gamma = 0.1876
        criterion = FocalLoss(gamma=gamma)


        # === Train ===
        final_train_acc, lr_history, train_loss_history, train_acc_history, train_soft_acc_history, train_f1_history = run_training(model, optimizer, scheduler, criterion, num_epochs, train_loader, device)
        print("Model trained!")

        # === Save the training outcomes ===
        filename = str(num_epochs)+'_iters_'+str(batch_size)+'batch_'+str(gamma)+'gamma_wo_dropout'
        file_path = f'UNet/Trials/{filename}.pkl'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data_tr_val = {
        "final_train_acc": final_train_acc,
        "lr_history": lr_history,
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "train_soft_acc_history": train_soft_acc_history,
        "train_f1_history": train_f1_history,
        }
        # Save the training and validating data
        with open(file_path, 'wb') as fp: 
                pickle.dump(data_tr_val, fp)

        # Save the model
        file_path = f'UNet/Models/{filename}.pth'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(model.state_dict(), file_path) 


        # === Training curves ===
        n_train = len(train_acc_history)
        t_train = num_epochs * np.arange(n_train) / n_train
        t_val = np.arange(1, num_epochs + 1)

        file_path = f'UNet/Plots/{filename}.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


if __name__ == '__main__':
        main()
