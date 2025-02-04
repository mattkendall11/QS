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


class Config2:
    """Configuration class for hyperparameters and model settings."""

    def __init__(self):
        # Model hyperparameters
        self.n_qubits = 8
        self.batch_size = 32
        self.cnn_channels = 64  # Replace LSTM hidden size with CNN channels
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


class qcnn(nn.Module):
    """Hybrid quantum-classical model combining 1D CNN and quantum circuit."""

    def __init__(self, input_dim: int, cnn_channels: int, n_qubits: int,
                 blocks: int, layers: int):
        super().__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Classical layers with batch normalization
        self.fc1 = nn.Linear(cnn_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)

        # Quantum layer
        qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Output layer
        self.fc5 = nn.Linear(n_qubits, 15)  # 15 = 5 horizons x 3 classes

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # CNN processing
        x = x.transpose(1, 2)
        # Change from (batch, seq_len, features) to (batch, features, seq_len)
        x = self.conv1(x)

        x = self.pool(x).squeeze(-1)  # Global average pooling

        x = self.bn1(nn.functional.relu(self.fc1(x)))

        # Quantum processing
        x = self.quantum_layer(x)

        logits = self.fc5(x)

        logits = logits.view(-1, 5, 3)

        if return_logits:
            return logits
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions

def test_one_pass():

    input_dim = 40
    config = Config2()
    model = qcnn(
            input_dim=input_dim,
        cnn_channels = config.cnn_channels,
        n_qubits = config.n_qubits,
    blocks = config.blocks,
        layers = config.layers
        )

    dummy_data = torch.rand(32,10,40)

    outut = model(dummy_data)
    return outut
test_one_pass()