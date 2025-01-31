import numpy as np
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


class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""

    def __init__(self, patience: int = 7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epochs, learning_rate) -> nn.Module:
    """Train the hybrid model with validation."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    best_model = None
    best_val_loss = float('inf')
    t_accuracy_vals = []
    v_accuracy_vals = []
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features, labels = batch
            features = features.float()

            optimizer.zero_grad()
            logits = model(features, return_logits=True)
            loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))

            loss.backward()
            optimizer.step()

            # Get predictions for accuracy calculation
            predictions = torch.argmax(logits, dim=-1)

            train_loss += loss.item()
            total += labels.size(0) * labels.size(1)  # account for all horizons
            correct += (predictions == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch
                features = features.float()

                logits = model(features, return_logits=True)
                loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))
                predictions = torch.argmax(logits, dim=-1)

                val_loss += loss.item()
                val_total += labels.size(0) * labels.size(1)
                val_correct += (predictions == labels).sum().item()

        # Print metrics
        logger.info(f'Epoch {epoch + 1}')
        logger.info(f'Train Loss: {train_loss / len(train_loader):.4f}')
        logger.info(f'Train Accuracy: {100 * correct / total:.2f}%')
        logger.info(f'Val Loss: {val_loss / len(val_loader):.4f}')
        logger.info(f'Val Accuracy: {100 * val_correct / val_total:.2f}%')
        t_accuracy_vals.append(100 * correct / total)
        v_accuracy_vals.append(100 * val_correct / val_total)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        # Early stopping check
        if early_stopping(val_loss):
            logger.info("Early stopping triggered")
            break

    return best_model, t_accuracy_vals, v_accuracy_vals


def plot_accuracies(tav, vav):
    x_axis = range(len(tav))
    plt.plot(x_axis, tav, label = 'Training accuracy')
    plt.plot(x_axis, vav, label = 'validation accuracy')
    plt.legend()
    plt.show()