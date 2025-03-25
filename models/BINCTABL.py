import logging
import torch
import torch.nn as nn

from utils.layers import BiN, BL_layer, TABL_layer
from use_case.dataset import FIDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import copy
from tqdm.auto import tqdm
import torch.optim as optim
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from use_case.metrics import compute_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinCTABL(nn.Module):
    def __init__(self, input_dim, return_logits = False):
        super().__init__()
        d1 = 40
        d2 = 60
        d3 = 120
        d4 = 3

        t1 = 10
        t2 = 10
        t3 = 5
        t4 = 1

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
        # x = torch.reshape(x,(x.shape[0],-1))
        #
        # logits = self.quantum_layer(x)
        # Process through each TABL head
        outputs = []
        for tabl in self.TABL_heads:
            self.max_norm_(tabl.W1.data)
            self.max_norm_(tabl.W.data)
            self.max_norm_(tabl.W2.data)
            out = tabl(x)
            out = torch.squeeze(out)
            out = torch.softmax(out, 1)
            outputs.append(out)

        # Stack the outputs for all horizons
        logits = torch.stack(outputs, dim =1)
        if return_logits:
            return logits
        else:
            predictions=torch.argmax(logits, dim=-1)
            return predictions  # Shape: (batch_size, num_horizons)

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))