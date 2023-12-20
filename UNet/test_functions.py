from train_functions import predict, accuracy
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_predictions(stimuli, model, device, test_loader, criterion, num=20):
    """
    Retrieves predictions and evaluation metrics for a given model over test data.

    Args:
    - stimuli (numpy.ndarray): Array of existing stimuli for correlation calculation.
    - model (torch.nn.Module): The UNet model for prediction.
    - device (torch.device): The device on which to perform computations (e.g., 'cuda' or 'cpu').
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - criterion: Loss function for model optimization (FocalLoss).
    - num (int, optional): Number of iterations for evaluation (default: 20).

    Returns:
    - points (list): List of dictionaries containing evaluation metrics for each iteration, i.e. loss, accuracy, balanced accuracy, F1 score.
    - losses (numpy.ndarray): Array of losses computed during evaluations.
    """

    model.eval()
    points = []
    losses = []
    for i, (data, target) in enumerate(test_loader):
      if i < num:
          data, target = data.to(device), target.to(device)
          output = model(data)

          loss = criterion(output, target)
          pred = predict(output)
          
          # Compute the most probable stimulus based on the correlation with the existing stimuli
          pred_temp = pred.cpu().view(5, -1).numpy()
          correlation_coefficients = [np.corrcoef(stimulus.flatten(), pred_temp.flatten())[0, 1] for stimulus in stimuli]
          closest = np.argmax(correlation_coefficients)
          pred = torch.from_numpy(stimuli[closest]).unsqueeze(0).to(device)

          acc = accuracy(pred, target)
          f1 = f1_score(target.view(-1).cpu(), pred.view(-1).cpu())
          soft_acc = balanced_accuracy_score(target.view(-1).cpu(), pred.view(-1).cpu())
          point = {'target': target.detach().cpu(),
          'predict': pred.detach().cpu(),
          'loss': loss.detach().cpu().numpy(),
            'accuracy': acc.detach().cpu().numpy(),
            'soft_accuracy': soft_acc,
                   'f1': f1,
          }
          losses.append(loss.detach().cpu().numpy())

          points.append(point)

    losses = np.array(losses)

    return points,losses


def visualize(points):
    """
    Visualizes target and predicted stimuli with corresponding loss.

    Args:
    - points (list): List of dictionaries containing 'target', 'predict', and 'loss' for 1 test example

    Returns:
    - None
    """
    
    for point in points:
        target = point['target']
        prediction = point['predict']
        loss = point['loss']

        target_stim = np.where(target == 1, 0.5, 0).squeeze()
        prediction_stim = np.where(prediction == 1, 0.5, 0).squeeze()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow(target_stim, cmap='gray')
        ax1.set_title('Target', fontsize=21)
        ax1.axis('off')

        ax2.imshow(prediction_stim, cmap='gray')
        ax2.set_title(f'Prediction, Loss: {np.round(loss,4)}', fontsize=21)
        ax2.axis('off')

        plt.show()
