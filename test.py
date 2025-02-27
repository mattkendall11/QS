from models.qLSTM import qlstm, Config
from models.qCNN import qcnn, Config2
from models.qbinc import qBinc
# from models.qnn import qnn
from utils.trainer import train_model, plot_accuracies
from utils.data_preprocessing import Big_data
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


learning_rate = 0.001
epochs = 400

def main():
    """Main execution function."""
    # Initialize configuration
    config = Config()

    # Load and split dataset
    train_dataset = FIDataset('test', 'data')
    val_dataset = FIDataset('val', 'data')
    test_dataset = FIDataset('test', 'data')
    all_labels =[]
    for _, label in train_dataset:
        all_labels.extend(label.flatten().tolist())
    label_counts = torch.bincount(torch.tensor(all_labels))

    weights = 1./label_counts.float()
    weights = weights/weights.sum()
    sample_weights = []
    for _, label in train_dataset:
        sample_weights.append(weights[label].mean())

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement = True
                                    )

    # turn sampler on ususally
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler = sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False)

    # Initialize model
    sample = next(iter(train_loader))
    features, label = sample
    input_dim = features.shape[2]
    # model = qBinc(input_dim = input_dim)
    # model = qnn()
    model = qlstm(
        input_dim=input_dim,
        lstm_hidden_size=config.lstm_hidden_size,
        n_qubits=config.n_qubits,
        blocks=config.blocks,
        layers=config.layers
    )



    # Train model
    trained_model,best_performer, tav, vav = train_model(model, train_loader, val_loader, learning_rate=learning_rate, epochs=epochs)

    # Save model
    torch.save(trained_model.state_dict(), fr'params/best_model_qlstm.pth')
    torch.save(best_performer.state_dict(), fr'params/best_performer_qlstm.pth')
    print("Training completed successfully")
    plot_accuracies(tav, vav)

    trained_model.eval()
    all_predictions = []
    all_targets = []
    for inputs, targets in tqdm(test_loader):
        predictions = model(inputs)
        all_predictions.append(predictions.detach().numpy())
        all_targets.append(targets.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    matches = np.sum(all_predictions == all_targets)
    total_elements = all_predictions.size
    percentage_match = (matches / total_elements) * 100

    print(f"Percentage of matching elements: {percentage_match:.2f}%, ratio = {matches}/{total_elements}")
    print(all_predictions, all_targets)
    print(compute_score(all_targets, all_predictions))


if __name__ == "__main__":
    main()

