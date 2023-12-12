from UNet import *
from Data_processing import get_dataloaders, get_data

def main() :
    param = {'num_epochs': num_epochs,
            'batch_size': batch_size,
            'Optimizer param lr': lr,
            'Optimizer param batas' : betas,
            'Optimizer param eps' : eps,
            'Optimizer param weight decay' : weight_decay,
            'scheduler param step size' : step_size,
            'scheduler param gamma': gamma}
    # ===== Data =====
    file_path = 'data/resampled_epochs_subj_0.pkl'
    path_to_save = 'UNet/trials'

    print("Getting data")
    epochs, labels = get_data(file_path, convention_neg=False, two_channels=False)
    print("Getting data_loaders")
    train_loader, val_loader, test_loader = get_dataloaders(epochs, labels, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=2).to(device)
    model = model.double()
    optimizer_kwargs = dict(
        lr=1e-3,
        weight_decay=1e-2,
        )
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
        )
    criterion = FocalLoss(gamma=gamma)
    # === Train === 
    print("Training !")
    final_train_acc, val_acc, val_soft_acc, lr_history, train_loss_history, train_acc_history, train_soft_acc_history, train_f1_history, val_loss_history, val_acc_history, val_soft_acc_history, val_f1_history = run_training(model, optimizer, scheduler, criterion, num_epochs, optimizer_kwargs, train_loader, val_loader, device)

if __name__ == "__main__":
    main()