import pennylane as qml
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from models.qLSTM import create_qnode


def plot_circuit(qnode, weight_shapes):
    """
    Plots the quantum circuit of a given QNode.

    Args:
        qnode (qml.QNode): The quantum node to visualize.
        weight_shapes (Dict): The dictionary containing the shapes of weights used in the circuit.
    """
    # Generate sample inputs and weights for visualization
    n_qubits = len(qnode.device.wires)
    inputs = [0.5] * n_qubits  # Example input values
    sample_weights = qml.numpy.random.rand(*weight_shapes["weights"])  # Random sample weights

    # Draw the circuit
    fig, ax = qml.draw_mpl(qnode)(inputs, sample_weights)
    plt.show()

# n_qubits, blocks, layers = 3, 2, 2
# qnode, weight_shapes = create_qnode(n_qubits, blocks, layers)
#
# plot_circuit(qnode, weight_shapes)