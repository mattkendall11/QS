import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from use_case.dataset import FIDataset

# Ensure reproducibility.
torch.manual_seed(42)
np.random.seed(42)

DEBUG_PRINT = False

# Dataset and DataLoader
class BINCTABL(nn.Module):
    def __init__(self, input_dim, num_classes, horizon_count):
        super(BINCTABL, self).__init__()

        # Feature Normalization
        self.feature_norm = nn.LayerNorm(input_dim)

        # Temporal Normalization
        self.temporal_norm = nn.LayerNorm(horizon_count)

        # Bilinear Layer
        self.bilinear_layer = nn.Bilinear(input_dim, horizon_count, 128)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Activation Functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input x: [batch_size, horizon_count, input_dim]

        # Feature normalization
        x = self.feature_norm(x)

        # Temporal normalization
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, horizon_count]
        x = self.temporal_norm(x)

        # Bilinear transformation
        horizon_count = x.size(-1)
        x = x.permute(0, 2, 1)  # Back to [batch_size, horizon_count, input_dim]
        x = torch.mean(x, dim=1)  # Averaging over the horizon_count dimension

        # Create second input for bilinear layer with matching batch size
        batch_size = x.size(0)
        bilinear_input2 = torch.arange(horizon_count).float().unsqueeze(0).repeat(batch_size, 1).to(x.device)
        x = self.bilinear_layer(x, bilinear_input2)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Configuration
input_dim = 40  # Features
num_classes = 3  # Upward, Stationary, Downward
horizon_count = 10  # Number of horizons

# Dataset preparation
dataset_type = "train"
dataset = FIDataset(dataset_type, 'data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = BINCTABL(input_dim=input_dim, num_classes=num_classes, horizon_count=horizon_count)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.float(), labels.long()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Train the model
train_model(model, dataloader, criterion, optimizer)