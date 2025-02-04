import logging
import torch
import torch.nn as nn
from utils.layers import BiN, BL_layer, TABL_layer
from tqdm.auto import tqdm
import torch.optim as optim
from typing import Tuple, Dict, List
import numpy as np
import pennylane as qml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



class qBinc(nn.Module):
    def __init__(self, input_dim, return_logits = False):
        super().__init__()
        d1 = 40
        d2 = 60
        d3 = 64
        d4 = 3

        t1 = 10
        t2 = 10
        t3 = 4
        t4 = 1

        n_qubits = 8
        blocks = 2
        layers = 2


        self.num_horizons = 5

        # Create separate TABL heads for each horizon
        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        # self.quantum_layer = QuantumLayer(n_qubits=4)
        # Create separate TABL layers for each horizon
        self.TABL_heads = nn.ModuleList([
            TABL_layer(d4, d3, t3, t4) for _ in range(self.num_horizons)
        ])
        qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc5 = nn.Linear(n_qubits, 15)  # 15 = 5 horizons x 3 classes
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, return_logits = False):
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x)

        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)

        self.max_norm_(self.BL2.W1.data)
        self.max_norm_(self.BL2.W2.data)

        x = self.BL2(x)
        x = self.dropout(x)

        x = torch.reshape(x, (x.shape[0], -1))


        x = self.quantum_layer(x)

        logits = self.fc5(x)

        logits = logits.view(-1, 5, 3)

        if return_logits:
            return logits
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))


def test_one_pass():

    input_dim = 40
    model = qBinc(input_dim=input_dim)

    dummy_data = torch.rand(32,10,40)
    model(dummy_data)
    outut = model(dummy_data)

    return outut
