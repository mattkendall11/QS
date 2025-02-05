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
from qiskit_ibm_runtime import QiskitRuntimeService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QiskitRuntimeService.save_account(channel="ibm_quantum",
                                  token="33e57983eeba524ab15fb723c217e66e8dc3e7c8cbcf348527036e97f5225c3c5efb683e671d6cd582fe177d1ac9085bdd865c87d848f39902b23cd8e7be1328",
                                  overwrite=True)
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=8)

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
    dev = qml.device("qiskit.remote", wires=n_qubits, backend = backend)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
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


class qnn(nn.Module):
    """Hybrid quantum-classical model combining LSTM and quantum circuit."""

    def __init__(self):
        super().__init__()
        n_qubits = 8
        blocks = 2
        layers = 2
        self.fc1 = nn.Linear(400, 256)
        qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Output layer
        self.fc5 = nn.Linear(n_qubits, 15)  # 15 = 5 horizons x 3 classes

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
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
    model = qnn()

    dummy_data = torch.rand(32,10,40)
    model(dummy_data)
    outut = model(dummy_data)

    return outut
output = test_one_pass()
print(output)