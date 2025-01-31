import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from use_case.dataset import FIDataset
from tqdm.auto import tqdm
import copy
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from use_case.metrics import compute_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for hyperparameters and model settings."""

    def __init__(self):
        # Model hyperparameters
        self.n_qubits = 8
        self.batch_size = 32
        self.lstm_hidden_size = 256
        self.blocks = 2
        self.layers = 2



def create_qnode(n_qubits: int, blocks: int, layers: int) -> Tuple[qml.QNode, Dict]:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):
        # Amplitude embedding for better quantum state preparation
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=True)

        # Multiple quantum processing blocks
        for i in range(blocks):
            qml.StronglyEntanglingLayers(weights[i], wires=range(n_qubits))

            # Add custom rotation gates for enhanced expressivity
            for j in range(n_qubits):
                qml.RY(weights[i][0][j][0], wires=j)
                qml.RZ(weights[i][0][j][1], wires=j)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (blocks, layers, n_qubits, 3)}
    return qnode, weight_shapes


class qlstm(nn.Module):
    """Hybrid quantum-classical model combining LSTM and quantum circuit."""

    def __init__(self, input_dim: int, lstm_hidden_size: int, n_qubits: int,
                 blocks: int, layers: int):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        lstm_output_size = lstm_hidden_size * 2  # Account for bidirectional

        # Classical layers with batch normalization
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.fc4 = nn.Linear(64, n_qubits)

        # Quantum layer
        qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Output layer
        self.fc5 = nn.Linear(n_qubits, 15)  # 15 = 5 horizons x 3 classes

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last timestep

        # Classical processing with batch normalization
        x = self.bn1(nn.functional.relu(self.fc1(x)))
        # x = self.bn2(nn.functional.relu(self.fc2(x)))
        # x = self.bn3(nn.functional.relu(self.fc3(x)))
        # x = nn.functional.relu(self.fc4(x))

        # Quantum processing
        x = self.quantum_layer(x)

        # Output processing
        logits = self.fc5(x)
        logits = logits.view(-1, 5, 3)

        if return_logits:
            return logits
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions