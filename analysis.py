import matplotlib.pyplot as plt

from models.qLSTM import qlstm, Config
from models.LSTM import LSTM
from models.qCNN import qcnn,Config2

import torch
from use_case.dataset import FIDataset
from use_case.metrics import compute_score, compute_metrics
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from utils.data_preprocessing import Big_data, cryptoDataset


# data = Big_data('data/BenchmarkDatasets', dataset_type='test', horizon=5, observation_length=10, train_val_split=0.8, n_trends=3)
data = FIDataset('val', 'data')
config = Config()
config2 = Config2()
'''
plot training val accuracies
'''
yt = np.load('params/training_accuracies.npy')
xt = range(len(yt))
yv = np.load('params/val_accuracies.npy')
plt.plot(xt,yt)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
test_loader = DataLoader(data, batch_size=config.batch_size,
                         shuffle=False, drop_last=False)


model1 = qlstm(
    input_dim=40,
    lstm_hidden_size=config.lstm_hidden_size,
    n_qubits=config.n_qubits,
    blocks=config.blocks,
    layers=config.layers
)
model2 = LSTM(input_dim=40,
             lstm_hidden_size=256,
             num_layers=2)
# model3 = qcnn(input_dim=400,
#               cnn_channels=config2.cnn_channels,
#               n_qubits=config2.n_qubits,
#               blocks=config2.blocks,
#               layers=config2.layers)
model1.load_state_dict(torch.load('params/best_performer_qlstm3.pth'))
model1.eval()
model2.load_state_dict(torch.load('params/best_performer_lstm.pth'))
model2.eval()
# model3.load_state_dict(torch.load('params/best_performer_qcnn.pth'))
# model3.eval()

all_predictionsq = []
all_targetsq = []
all_predictions = []
all_targets = []
all_predictionsc = []
all_targetsc = []
for batch_inputs, batch_targets in tqdm(test_loader):
    # Generate predictions
    batch_predictions = model1(batch_inputs)

    all_predictionsq.append(batch_predictions.cpu().numpy())
    all_targetsq.append(batch_targets.numpy())

    batch_predictions = model2(batch_inputs)

    all_predictions.append(batch_predictions.cpu().numpy())
    all_targets.append(batch_targets.numpy())

    # batch_predictions = model3(batch_inputs)
    #
    # all_predictionsc.append(batch_predictions.cpu().numpy())
    # all_targetsc.append(batch_targets.numpy())

# Concatenate all predictions and targets
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
all_predictionsq = np.concatenate(all_predictionsq, axis=0)
all_targetsq = np.concatenate(all_targetsq, axis=0)
# all_predictionsc = np.concatenate(all_predictionsc, axis=0)
# all_targetsc = np.concatenate(all_targetsc, axis=0)
# matches = np.sum(all_predictions == all_targets)
# total_elements = all_predictions.size
# percentage_match = (matches / total_elements) * 100
#
# print(f"Percentage of matching elements: {percentage_match:.2f}%, ratio = {matches}/{total_elements}")
# Compute the overall score
score = compute_score(all_targets, all_predictions)
scoreq = compute_score(all_targetsq, all_predictionsq)
# scorec = compute_score(all_targetsc, all_predictionsc)
print(score,scoreq)

