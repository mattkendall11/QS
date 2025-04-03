from torch.backends.cudnn import benchmark

from models.qLSTM import qlstm, Config, create_qnode
import torch
from torch.utils.data import DataLoader
from utils.visualise_circuit import plot_circuit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from utils.data_preprocessing import Big_data

# Load dataset
data = Big_data('data/BenchmarkDatasets', dataset_type='test', horizon=5, observation_length=10, train_val_split=0.8, n_trends=3)
config = Config()

test_loader = DataLoader(data, batch_size=config.batch_size, shuffle=False, drop_last=False)

# Initialize model
sample = next(iter(test_loader))
features, label = sample
input_dim = features.shape[2]

model = qlstm(
    input_dim=40,
    lstm_hidden_size=config.lstm_hidden_size,
    n_qubits=config.n_qubits,
    blocks=config.blocks,
    layers=config.layers
)
model.load_state_dict(torch.load('params/best_performer_qlstm.pth'))
model.eval()


# all_predictions = []
# all_targets = []
#
# # Generate predictions
# for batch_inputs, batch_targets in tqdm(test_loader):
#     batch_predictions = model(batch_inputs).detach().cpu().numpy()
#
#     all_predictions.append(batch_predictions)
#     all_targets.append(batch_targets.numpy())
#
# # Concatenate all predictions and targets
# all_predictions = np.concatenate(all_predictions, axis=0)
#
# all_targets = np.concatenate(all_targets, axis=0)
# np.save('params/all_predictions.npy', all_predictions)
# np.save('params/all_targets.npy', all_targets)
# Define class labels for stock movement

all_predictions = np.load('params/all_predictions.npy')
all_targets = np.load('params/all_targets.npy')
class_labels = ['Down', 'Stationary', 'Up']

horizons = ['1', '2', '3', '5', '10']
accuracies = []
# Iterate over all five horizons and plot confusion matrices
# for horizon in range(5):
#
#     # Extract predictions and targets for this horizon
#     horizon_predictions = all_predictions[:, horizon]
#     horizon_targets = all_targets[:, horizon]
#     matches = np.sum(horizon_predictions == horizon_targets)
#     total_elements = len(horizon_predictions)
#     percentage_match = (matches / total_elements) * 100
#     accuracies.append(percentage_match)
#     # Compute confusion matrix
#     conf_matrix = confusion_matrix(horizon_targets, horizon_predictions, normalize='all')
#
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=class_labels,
#                 yticklabels=class_labels, cbar=False)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix for Horizon {horizons[horizon]}')
#     plt.show()
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1 row, 5 columns for the subplots
axes = axes.flatten()
for horizon in range(5):
    # Extract predictions and targets for this horizon
    horizon_predictions = all_predictions[:, horizon]
    horizon_targets = all_targets[:, horizon]
    matches = np.sum(horizon_predictions == horizon_targets)
    total_elements = len(horizon_predictions)
    percentage_match = (matches / total_elements) * 100
    accuracies.append(percentage_match)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(horizon_targets, horizon_predictions, normalize='true')

    # Plot confusion matrix on the corresponding axis
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels, cbar=False, ax=axes[horizon])
    axes[horizon].set_xlabel('Predicted')
    axes[horizon].set_ylabel('Actual')
    axes[horizon].set_title(f'Horizon {horizons[horizon]}')

plt.tight_layout()
plt.savefig('cm.svg')
plt.show()


lstm = [91.34751559, 92.15082538, 93.67199561, 94.12894169, 93.71588446]
qlstm = [95.18980238, 95.01561763, 95.76465133, 95.86619741, 95.53567548]
qcnn = []
benchmark = np.array([88.7, 80.6, 80.1, 88.2, 91.6])
plt.plot(horizons, accuracies, color = 'b')
plt.plot(horizons, accuracies, '+', label = 'q-lstm', color = 'b')
plt.plot(horizons, lstm, color = 'darkblue')
plt.plot(horizons, lstm, '+', label = 'lstm', color = 'darkblue')
plt.plot(horizons, benchmark, color = 'r')
plt.plot(horizons, benchmark,'+', color = 'r', label = 'Benchmark')
plt.legend()
plt.xlabel('Horizon')
plt.ylabel('F1 score')
plt.savefig('comparison.svg')
plt.show()