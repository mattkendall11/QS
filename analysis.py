from models.qLSTM import qlstm, Config
import torch
from use_case.dataset import FIDataset
from use_case.metrics import compute_score, compute_metrics
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
config = Config()


test_dataset = FIDataset('train', 'data')


test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                         shuffle=False, drop_last=False)

# Initialize model
sample = next(iter(test_loader))
features, label = sample
input_dim = features.shape[2]

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

# Compute the overall score
score = compute_score(all_targets, all_predictions)
print(score)

