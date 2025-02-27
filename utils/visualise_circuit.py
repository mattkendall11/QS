import pennylane as qml
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import torch
from models.qLSTM import create_qnode


def plot_circuit(qnode, weights):
    """
    Draws the quantum circuit for a given QNode and its associated weights.

    Args:
        qnode (qml.QNode): The quantum node representing the quantum circuit.
        weights (tensor or dict): The weights (parameters) for the quantum circuit.
    """
    # Generate sample inputs for the circuit (this could be customized)
    n_qubits = len(qnode.device.wires)
    inputs = [0.5] * n_qubits  # Example input values (you can change this if needed)

    # Draw the circuit using matplotlib and the given weights
    fig, ax = qml.draw_mpl(qnode)(inputs, weights)

    # Display the quantum circuit plot
    plt.show()
# n_qubits, blocks, layers = 3, 2, 2
# qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
#
# plot_circuit(qnode, weight_shapes)