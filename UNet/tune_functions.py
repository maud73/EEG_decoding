import torch
import optuna
from train_functions import predict
from model import UNet
from loss import FocalLoss
from sklearn.metrics import f1_score

def test_f1(model, val_loader_hyp, device):
    f1 = 0
    with torch.no_grad():
        for data, target in val_loader_hyp:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = predict(output)
            f1 += f1_score(target.view(-1).cpu(), pred.view(-1).cpu())
    f1 /= len(val_loader_hyp.dataset)

    return f1

# Edited from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py and https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py

def objective(trial, device, train_loader_hyp, val_loader_hyp, num_epochs):
    """Objective function to be optimized by Optuna

    Hyperparameters to be optimized: learning algorithm, learning rate, weight decay,
    betas (control the exponential decay rate of the running mean and variance), focal parameter of the loss

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - f1 score (float): The test F1 score. Parameter to be maximized.
    """

    # Generate the model
    model = UNet(n_channels=1, n_classes=2).to(device)
    model = model.double()
    model = model.to(device)

    # Generate the optimizers for: the optimizer, the criterion, and the learning rate scheduler
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.9, 0.999, log=True)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    focus = trial.suggest_float('focus', 1e-9, 5, log=True)
    criterion = FocalLoss(gamma=focus)

    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR'])
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_float('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        eta_min = trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(len(train_loader_hyp.dataset) * num_epochs) // train_loader_hyp.batch_size,
            eta_min=eta_min
        )

    # Train and validate the model
    for epoch in range(1, num_epochs + 1):
      for batch_idx, (data,target) in enumerate(train_loader_hyp) :
        model.train()
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

        model.eval()
        f1 = test_f1(model, val_loader_hyp, device)

    # Pruning
    trial.report(f1, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return f1