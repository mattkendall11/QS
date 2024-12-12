import numpy as np
import torch
import torch.nn as nn
from use_case.run import BatchType, OutputType


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        num_horizons = 5
        # Generate random predictions for each sample and horizon
        random_predictions = torch.randint(0, 3, (batch_size, num_horizons))
        return random_predictions


async def solve(input: BatchType) -> OutputType:
    # Ensure reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    # Create our model
    model = DummyModel()
    # Get predictions from the model
    predictions = model(torch.Tensor(input[0]))
    # Move predictions to CPU and convert to NumPy array
    return predictions.cpu().numpy()
