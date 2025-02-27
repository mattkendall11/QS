import torch
import torch.nn as nn
import hiddenlayer as hl
import pennylane as qml
from models.qLSTM import qlstm, Config
import torch.onnx
config = Config()

model = qlstm(
    input_dim=40,
    lstm_hidden_size=config.lstm_hidden_size,
    n_qubits=config.n_qubits,
    blocks=config.blocks,
    layers=config.layers
)

dummy_data = torch.rand(32, 10, 40)

torch.onnx.export(model, dummy_data, 'model.onxx', export_params=True)
