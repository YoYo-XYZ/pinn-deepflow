import pennylane as qml
import pennylane.numpy as np   # or import numpy as np if you prefer standard NumPy

# 5-qubit device (you can change to any device, e.g. 'lightning.qubit' for speed)
dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuit(features, weights):
    """
    Replicates the exact circuit shown in the diagram.
    
    - features: array of shape (5,) → data for AngleEmbedding
    - weights: array of shape (3, 5) → variational parameters
        weights[0] → angles for the first RX layer
        weights[1] → angles for the RZ layer
        weights[2] → angles for the final RX layer
    """
    
    # ================== LEFT PART ==================
    # Tall AngleEmbedding block (exactly as drawn)
    qml.AngleEmbedding(features=features, wires=range(5))
    
    # RX + RZ on every qubit (right after the embedding block)
    for i in range(5):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    
    # Blue barrier (optional for execution, required for drawing to match the diagram)
    qml.Barrier()
    
    # ================== RIGHT PART ==================
    # Final RX layer on every qubit (the labeled "RX" boxes)
    for i in range(5):
        qml.RX(weights[2, i], wires=i)
    
    # Black vertical line with dots is interpreted as a second barrier
    # (common in many circuit-drawing tools; it separates layers visually)
    qml.Barrier()
    
    # Measurements / observables on every qubit
    # (the rightmost boxes with the arrow are the measurement symbols)
    return [qml.expval(qml.Z(i)) for i in range(5)]


# ====================== EXAMPLE USAGE ======================

# Dummy data (5 features for 5 qubits)
features = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

# Dummy variational weights (3 layers × 5 qubits)
weights = np.random.random((3, 5))

# Run the circuit
result = circuit(features, weights)
print("Expectation values:", result)

# ====================== VISUALISE IT (matches the diagram) ======================

# Text drawing (very close to your diagram)
print(qml.draw(circuit, level="device")(features, weights))

# Or nice matplotlib version (recommended)
fig, ax = qml.draw_mpl(circuit, style="pennylane")(features, weights)
# fig.savefig("my_circuit.png")   # save as image if you want