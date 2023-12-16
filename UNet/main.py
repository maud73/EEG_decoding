from Data_processing import get_dataloaders, get_data
from train_functions import run_training
from Random import set_random_seeds

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import UNet
from loss import FocalLoss
from test import test
import os 
import pickle


def main():
        print("UNet")
        # === Set random seeds for reproducibility ===
        set_random_seeds()
        # === Load the data ===        
        file_path = 'data/resampled_epochs_subj_0.pkl'
        epochs, labels = get_data(file_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epochs = 2 #to change for debugging
        batch_size = 8 #to change for debugging
        data_kwargs = dict(
        epochs=epochs[:155], #to change for debugging
        labels=labels[:155], #to change for debugging
        batch_size=batch_size
        )
        
        train_loader, val_loader, test_loader = get_dataloaders(**data_kwargs)
        print("Data loaded")

        # === Define the UNet and the training hyperparameters ===
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
        gamma = 0.1876 # To be tuned
        criterion = FocalLoss(gamma=gamma)


        # === Train ===
        final_train_acc, val_acc, val_soft_acc, lr_history, train_loss_history, train_acc_history, train_soft_acc_history, train_f1_history, val_loss_history, val_acc_history, val_soft_acc_history, val_f1_history = run_training(model, optimizer, scheduler, criterion, num_epochs, optimizer_kwargs, train_loader, val_loader, device)
        print("Model trained!")

        # === Save the training outcomes ===
        filename = str(num_epochs)+'_iters_'+str(batch_size)+'batch_'+str(gamma)+'gamma'
        file_path = f'UNet/Trials/{filename}.pkl'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data_tr_val = {
        "final_train_acc": final_train_acc,
        "val_acc": val_acc,
        "val_soft_acc": val_soft_acc,
        "lr_history": lr_history,
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "train_soft_acc_history": train_soft_acc_history,
        "train_f1_history": train_f1_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "val_soft_acc_history": val_soft_acc_history,
        "val_f1_history": val_f1_history,
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

        plt.figure(figsize=(6.4, 4.8*5))
        plt.subplot(5, 1, 2)
        plt.plot(t_train, train_acc_history, label="Train")
        plt.plot(t_val, val_acc_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(5, 1, 4)
        plt.plot(t_train, train_f1_history, label="Train")
        plt.plot(t_val, val_f1_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("F1 score")

        plt.subplot(5, 1, 1)
        plt.plot(t_train, train_loss_history, label="Train")
        plt.plot(t_val, val_loss_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(5, 1, 5)
        plt.plot(t_train, lr_history)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        plt.subplot(5, 1, 3)
        plt.plot(t_train, train_soft_acc_history, label="Train")
        plt.plot(t_val, val_soft_acc_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Soft accuracy")
        plt.savefig(file_path)


        # === Test the best model ===
        file_path = f'UNet/Plots/Tests/{filename}.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        test(best_model=model, test_loader=test_loader, file_path=file_path)

if __name__ == '__main__':
        main()
