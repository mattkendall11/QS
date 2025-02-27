from models.qLSTM import qlstm, Config
import torch
from use_case.dataset import FIDataset
from use_case.metrics import compute_score, compute_metrics
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from utils.data_preprocessing import Big_data


data = Big_data('data/BenchmarkDatasets', dataset_type='val', horizon=5, observation_length=10, train_val_split=0.8, n_trends=3)
config = Config()


test_loader = DataLoader(data, batch_size=config.batch_size,
                         shuffle=False, drop_last=False)

# Initialize model
sample = next(iter(test_loader))
features, label = sample
input_dim = features.shape[2]
print(features.shape)
print(label.shape)
model = qlstm(
    input_dim=input_dim,
    lstm_hidden_size=config.lstm_hidden_size,
    n_qubits=config.n_qubits,
    blocks=config.blocks,
    layers=config.layers
)
model.load_state_dict(torch.load('params/best_performer_qlstm.pth'))
model.eval()


all_predictions = []
all_targets = []

for batch_inputs, batch_targets in tqdm(test_loader):
    # Generate predictions
    batch_predictions = model(batch_inputs)

    all_predictions.append(batch_predictions.cpu().numpy())
    all_targets.append(batch_targets.numpy())

# Concatenate all predictions and targets
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

matches = np.sum(all_predictions == all_targets)
total_elements = all_predictions.size
percentage_match = (matches / total_elements) * 100

print(f"Percentage of matching elements: {percentage_match:.2f}%, ratio = {matches}/{total_elements}")
# Compute the overall score
score = compute_score(all_targets, all_predictions)
print(score)

