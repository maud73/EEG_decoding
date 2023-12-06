import torch
from model import UNet
from loss import FocalLoss
from UNet.train_functions import run_training
import numpy as np
import matplotlib.pyplot as plt
import pickle
from .. import Data_processing


# === Load the data ===
epochs, labels = Data_processing.get_data(file_path='data/resampled_epochs_subj_0.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 2
batch_size = 30
data_kwargs = dict(
    epochs=epochs,
    labels=labels,
    batch_size=batch_size
)
train_loader, val_loader, test_loader = Data_processing.get_dataloaders(**data_kwargs)


# === Define the UNet and the training parameters ===
model = UNet(n_channels=1, n_classes=2).to(device)
model = model.double()
criterion = FocalLoss(gamma=1) # Gamma to be tuned
optimizer_kwargs = dict(
    lr=1e-3,
    weight_decay=1e-2,
)
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
)

# === Train ===
final_train_acc, val_acc, lr_history, train_loss_history, train_acc_history, train_f1_history, val_loss_history, val_acc_history, val_f1_history = run_training(model, optimizer, scheduler, criterion, num_epochs, optimizer_kwargs, train_loader, val_loader, device)


# === Save the training outcomes ===
data = {
    "final_train_acc": final_train_acc,
    "val_acc": val_acc,
    "lr_history": lr_history,
    "train_loss_history": train_loss_history,
    "train_acc_history": train_acc_history,
    "train_f1_history": train_f1_history,
    "val_loss_history": val_loss_history,
    "val_acc_history": val_acc_history,
    "val_f1_history": val_f1_history,
}

with open('training.pkl', 'wb') as file:
    pickle.dump(data, file)


# ===== Plot training curves =====
n_train = len(train_acc_history)
t_train = num_epochs * np.arange(n_train) / n_train
t_val = np.arange(1, num_epochs + 1)

plt.figure(figsize=(6.4 * 4, 4.8))
plt.subplot(1, 3, 1)
plt.plot(t_train, train_acc_history, label="Train")
plt.plot(t_val, val_acc_history, label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")

plt.subplot(1, 3, 4)
plt.plot(t_train, train_f1_history, label="Train")
plt.plot(t_val, val_f1_history, label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("F1 score")
plt.savefig("f1_score.png")

plt.subplot(1, 3, 2)
plt.plot(t_train, train_loss_history, label="Train")
plt.plot(t_val, val_loss_history, label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

plt.subplot(1, 3, 3)
plt.plot(t_train, lr_history)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.savefig("lr.png")

# ===== Plot low/high loss predictions on validation set =====
points = get_predictions(
    model,
    device,
    val_loader,
    criterion,
)
points.sort(key=lambda x: x[1])
plt.figure(figsize=(15, 6))
for k in range(5):
    plt.subplot(2, 5, k + 1)
    plt.imshow(points[k][0][0, 0], cmap="gray")
    plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
    plt.subplot(2, 5, 5 + k + 1)
    plt.imshow(points[-k - 1][0][0, 0], cmap="gray")
    plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")