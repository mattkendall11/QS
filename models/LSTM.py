import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm.auto import tqdm
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config_lstm:
    """Configuration class for hyperparameters and model settings."""

    def __init__(self):
        # Model hyperparameters
        self.batch_size = 32
        self.input_dim = 40
        self.lstm_hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.2
        self.bidirectional = True


class LSTM(nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_size: int, num_layers: int,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output size (depends on bidirectional setting)
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # Output layer for 5 horizons with 3 classes each
        self.fc_out = nn.Linear(128, 15)  # 15 = 5 horizons x 3 classes

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last timestep output

        # Fully connected layers
        x = self.bn1(nn.functional.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(nn.functional.relu(self.fc2(x)))

        # Output layer
        logits = self.fc_out(x)
        logits = logits.view(-1, 5, 3)  # Reshape to (batch_size, 5 horizons, 3 classes)

        if return_logits:
            return logits
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions


def test_one_pass():
    config = Config_lstm()
    model = LSTM(
        input_dim=config.input_dim,
        lstm_hidden_size=config.lstm_hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    )

    # Create dummy batch of data: [batch_size, sequence_length, input_features]
    dummy_data = torch.rand(32, 10, config.input_dim)

    # Forward pass
    output = model(dummy_data)
    print(f"Output shape: {output.shape}")  # Should be [32, 5]
    return output
