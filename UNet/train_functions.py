import torch
from sklearn.metrics import f1_score


def predict(output):
    """
    Predicts each pixel's class (i.e. the decoded visual stimulus) from the UNet's output

    Args:
        output (Tensor): n_batch, depth 2, height and width 5 (output of the U-Net)

    Returns:
        pred (Tensor): n_batch, height and width 5
    """
    pred = torch.argmax(output,dim=1)
    return pred


def accuracy(prediction, target):
    """
    Computes the accuracy of the prediction

    Args:
        prediction (Tensor): predicted stimulus (n_batch, height and width 5)
        target (Tensor): target stimulus (n_batch, height and width 5)

    Returns:
        accuracy (float)
    """
    return torch.mean((prediction == target).float())


def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device, train_only_one=False):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.WeightedFocalLoss
    @param dataset_train: torch.utils.data.DataLoader
    @param dataset_test: torch.utils.data.DataLoader
    @param optimizer: torch.optim.Optimizer (AdamW)
    @param scheduler: torch.optim.lr_scheduler (CosineAnnealingLR(optimizer, T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,)
    @device:
    """
    model.train()
    loss_history = []
    accuracy_history = []
    f1_history = []
    lr_history = []

    for batch_idx, (data,target) in enumerate(train_loader) :
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Compute the gradient
        loss = criterion(output,target)
        loss.backward()

        # Update the parameters of the model with a gradient step
        optimizer.step()
        scheduler.step()

        pred = predict(output)
        acc = accuracy(pred, target)
        f1 = f1_score(target.view(-1).cpu(), pred.view(-1).cpu())

        loss_float = loss.item()
        loss_history.append(loss_float)
        accuracy_history.append(acc)
        f1_history.append(f1)

        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={acc:0.3f} "
                f"batch_f1={f1:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )

    return loss_history, accuracy_history, f1_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion, train_only_one):
    model.eval()
    val_loss = 0
    correct = 0
    f1 = 0

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += criterion(output, target).item() * len(data)
        pred = predict(output)
        correct += pred.eq(target.view_as(pred)).sum().item()
        f1 += f1_score(target.view(-1).cpu(), pred.view(-1).cpu())

    f1 /= len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(
        "Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 score: {}".format(
            val_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
            f1,
        )
    )
    return val_loss, correct / len(val_loader.dataset), f1

def run_training(
    model,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    optimizer_kwargs,
    train_loader,
    val_loader,
    device="cuda",
    train_only_one=False
):

    # ===== Model =====
    model = model.to(device=device)

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    train_f1_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device, train_only_one
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        train_f1_history.extend(train_f1)
        lr_history.extend(lrs)

        val_loss, val_acc, val_f1 = validate(model, device, val_loader, criterion, train_only_one)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)

    return sum(train_acc) / len(train_acc), val_acc, lr_history, train_loss_history, train_acc_history, train_f1_history, val_loss_history, val_acc_history, val_f1_history