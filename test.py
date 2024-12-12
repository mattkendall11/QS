import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from use_case.dataset import FIDataset

# Ensure reproducibility.
torch.manual_seed(42)
np.random.seed(42)
DEBUG_PRINT=False

dataset_type = "train"
dataset = FIDataset(dataset_type, 'data')

