"""Optional quantum-enhanced PINN architectures."""

from typing import List, Optional, Union, Callable
from .nn import NN
from .utility import get_dtype

from torch import nn

import pennylane as qml


# ---------------------------------------------------------------------------
# Module-level quantum circuits for QCPINN (must be at module scope for
# pickling / deepcopy / multiprocessing compatibility).
# ---------------------------------------------------------------------------

def _qcpinn_circuit(inputs, weights):
    """Published QCPINN cascade circuit.

    This is the cascade ansatz from the QCPINN reference implementation:
    X-angle embedding, RX and RZ rotations on every qubit, followed by a
    trainable CRX ring.  The three rows of ``weights`` correspond to RX, RZ,
    and CRX parameters respectively.

    Number of qubits and layer count are inferred from the *runtime shapes* of the
    tensors so that no closure over instance state is needed and the
    function remains picklable.
    """
    nqubits = inputs.shape[-1]
    n_layers = weights.shape[0]

    qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")

    for layer in range(n_layers):
        for i in range(nqubits):
            qml.RX(weights[layer, 0, i], wires=i)
        for i in range(nqubits):
            qml.RZ(weights[layer, 1, i], wires=i)
        qml.CRX(weights[layer, 2, 0], wires=[nqubits - 1, 0])
        for i in range(1, nqubits):
            qml.CRX(weights[layer, 2, i], wires=[i - 1, i])

    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]


def _qcpinn_hea_circuit(inputs, weights):
    """Pictured HEA circuit with fixed linear nearest-neighbor CNOTs.

    Each layer applies trainable RX, RY, and RZ rotations to every qubit,
    followed by CNOTs from qubit ``i`` to ``i + 1`` for ``i < nqubits - 1``.
    The angle embedding is retained so this mode is a drop-in QCPINN
    alternative; the HEA topology refers to the trainable ansatz block.
    """
    nqubits = inputs.shape[-1]
    n_layers = weights.shape[0]

    qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")

    for layer in range(n_layers):
        for i in range(nqubits):
            qml.RX(weights[layer, 0, i], wires=i)
            qml.RY(weights[layer, 1, i], wires=i)
            qml.RZ(weights[layer, 2, i], wires=i)
        for i in range(nqubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]


_QCPINN_LAYER_TYPES = {
    "cascade": _qcpinn_circuit,
    "hea": _qcpinn_hea_circuit,
}

class QCPINN(NN):
    """Quantum-circuit PINN with a configurable trainable circuit block.

    Args:
        input_vars: Names of coordinate inputs.
        output_vars: Names of predicted fields.
        nqubits: Number of qubits in the circuit.
        q_layer_type: Circuit topology: ``"cascade"`` or ``"hea"``.
        q_layer_iterations: Number of repeated quantum layers.
        hidden_layer_pre: Classical layer widths before the quantum layer.
        hidden_layer_post: Classical layer widths after the quantum layer.
        activation: PyTorch activation module for classical layers.
        weight_init: Weight initialization scheme inherited from ``NN``.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        nqubits: Optional[int] = 4,
        q_layer_type: str = "cascade",
        q_layer_iterations: int = 1,
        hidden_layer_pre: Optional[List[int]] = None,
        hidden_layer_post: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """Initialize and build the quantum-circuit PINN."""
        super().__init__(input_vars, output_vars, weight_init=weight_init)
        self.nqubits = nqubits
        self.hidden_layer_pre = hidden_layer_pre if hidden_layer_pre is not None else []
        self.hidden_layer_post = hidden_layer_post if hidden_layer_post is not None else []
        self.activation = activation
        if not isinstance(q_layer_type, str) or q_layer_type.lower() not in _QCPINN_LAYER_TYPES:
            supported = ", ".join(sorted(_QCPINN_LAYER_TYPES))
            raise ValueError(
                f"Unsupported q_layer_type={q_layer_type!r}. "
                f"Expected one of: {supported}."
            )
        self.q_layer_type = q_layer_type.lower()
        self.q_layer_iterations = q_layer_iterations
        self._build_network()
        self.to(get_dtype())
    
    def _qnn_setup(self):
        qml_device = qml.device("default.qubit", wires=self.nqubits)

        qnode = qml.QNode(
            _QCPINN_LAYER_TYPES[self.q_layer_type],
            qml_device,
            interface="torch",
        )
        qlayer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"weights": (self.q_layer_iterations, 3, self.nqubits)},
        )
        return qlayer

    def _build_network(self):
        layers = []

        # Pre-processing layers
        iter_layers = [self.input_num] + self.hidden_layer_pre + [self.nqubits]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        q_layer = self._qnn_setup()
        # Quantum layers
        layers.append(q_layer)
        layers.append(self.activation)
        
        # Post-processing layers
        iter_layers = [self.nqubits] + self.hidden_layer_post + [self.output_num]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

class QPINN(NN):
    """Quantum PINN with configurable feature map and ansatz layers.

    Args:
        input_vars: Names of coordinate inputs.
        output_vars: Names of predicted fields.
        nqubits: Number of qubits in the circuit.
        q_depth: Number of trainable quantum layers.
        feature_map: Feature map: ``"simple"``, ``"product"``, or
            ``"chebyshev"``.
        ansatz: Trainable circuit block: ``"simple"``, ``"hea"``, or
            ``"cascade"``.
        cost_func: Measurement block. Only ``"simple"`` is currently
            supported.
        hidden_layer_pre: Classical layer widths before the quantum layer.
        hidden_layer_post: Classical layer widths after the quantum layer.
        activation: PyTorch activation module for classical layers.
        weight_init: Weight initialization scheme inherited from ``NN``.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        nqubits: Optional[int] = 4,
        q_depth: int = 4,
        feature_map: str = "simple",
        ansatz: str = "hea",
        cost_func: str = "simple",
        hidden_layer_pre: Optional[List[int]] = None,
        hidden_layer_post: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """Initialize and build the quantum PINN."""
        super().__init__(input_vars, output_vars, weight_init=weight_init)
        if isinstance(nqubits, bool) or not isinstance(nqubits, int) or nqubits < 1:
            raise ValueError("nqubits must be a positive integer")
        if isinstance(q_depth, bool) or not isinstance(q_depth, int) or q_depth < 1:
            raise ValueError("q_depth must be a positive integer")
        if feature_map not in {"simple", "product", "chebyshev"}:
            raise ValueError(
                "Unsupported feature_map. Expected one of: "
                "simple, product, chebyshev."
            )
        if ansatz not in {"simple", "hea", "cascade"}:
            raise ValueError(
                "Unsupported ansatz. Expected one of: simple, hea, cascade."
            )
        if cost_func != "simple":
            raise ValueError("Unsupported cost_func. Expected: simple.")
        self.nqubits = nqubits
        self.q_depth = q_depth
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.cost_func = cost_func
        self.hidden_layer_pre = hidden_layer_pre if hidden_layer_pre is not None else []
        self.hidden_layer_post = hidden_layer_post if hidden_layer_post is not None else []
        self.activation = activation
        self._build_network()
        self.to(get_dtype())

    def _qnn_layer(self):
        qml_device = qml.device("default.qubit", wires=self.nqubits)

        feature_map = self.feature_map
        ansatz = self.ansatz
        nqubits = self.nqubits
        q_depth = self.q_depth

        @qml.qnode(qml_device, interface="torch")
        def qnn_layer(inputs, weights):
            if feature_map == "product":
                embedded_inputs = qml.math.asin(
                    qml.math.clip(inputs, -1.0, 1.0)
                )
            elif feature_map == "chebyshev":
                clipped_inputs = qml.math.clip(inputs, -1.0, 1.0)
                embedded_inputs = 2.0 * clipped_inputs.square() - 1.0
            else:
                embedded_inputs = inputs
            qml.AngleEmbedding(
                embedded_inputs,
                wires=range(nqubits),
                rotation="Y",
            )

            for layer in range(q_depth):
                if ansatz == "simple":
                    for wire in range(nqubits):
                        qml.RY(weights[layer, 0, wire], wires=wire)
                elif ansatz == "hea":
                    for wire in range(nqubits):
                        qml.RX(weights[layer, 0, wire], wires=wire)
                        qml.RY(weights[layer, 1, wire], wires=wire)
                        qml.RZ(weights[layer, 2, wire], wires=wire)
                    for wire in range(nqubits - 1):
                        qml.CNOT(wires=[wire, wire + 1])
                else:
                    for wire in range(nqubits):
                        qml.RX(weights[layer, 0, wire], wires=wire)
                        qml.RY(weights[layer, 1, wire], wires=wire)
                    if nqubits > 1:
                        for wire in range(nqubits):
                            qml.CRX(
                                weights[layer, 2, wire],
                                wires=[wire, (wire - 1) % nqubits],
                            )

            return [qml.expval(qml.PauliZ(wire)) for wire in range(nqubits)]

        return qnn_layer

    def _build_network(self):
        layers = []

        # Pre-processing layers
        iter_layers = [self.input_num] + self.hidden_layer_pre + [self.nqubits]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        # Quantum layers
        q_layer = qml.qnn.TorchLayer(
            self._qnn_layer(),
            weight_shapes={"weights": (self.q_depth, 3, self.nqubits)},
        )
        layers.append(q_layer)
        layers.append(self.activation)
        
        # Post-processing layers
        iter_layers = [self.nqubits] + self.hidden_layer_post + [self.output_num]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
