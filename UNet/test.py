import torch
from train_functions import *
import matplotlib.pyplot as plt

@torch.no_grad()
def test(best_model, test_loader, file_path):
    soft_acc_test_history = torch.zeros(1,5,5)
    for data, target in test_loader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = best_model(data)
        pred = predict(output)
        soft_acc = soft_accuracy(pred, target, reduction='none')
        # f1 = f1_score(target.view(-1).cpu(), pred.view(-1).cpu())
        soft_acc_test_history += soft_acc

    # === Plot pixel-wise mean accuracy and F1 score ===
    soft_acc_test_mean = (soft_acc_test_history / len(test_loader)).numpy().squeeze()
    background_intensity = np.mean(soft_acc_test_mean)
    for i in range(soft_acc_test_mean.shape[0]):
        for j in range(soft_acc_test_mean.shape[1]):
            text_color = 'black' if soft_acc_test_mean[i, j] < background_intensity else 'white'
            plt.text(j, i, f'{soft_acc_test_mean[i, j]:.2f}', ha='center', va='center', color=text_color)
    plt.imshow(soft_acc_test_mean, cmap='YlGn', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Pixel-wise accuracy', rotation=270, labelpad=15)
    plt.savefig(file_path)
