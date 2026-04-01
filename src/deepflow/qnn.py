from typing import List, Optional
from .nn import NN
import platform

from torch import nn

import pennylane as qml
class QPINN(NN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        nqubits: Optional[int] = 4,
        q_depth: int = 4,
        hidden_layer_pre: Optional[List[int]] = None,
        hidden_layer_post: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__(input_vars, output_vars)
        self.nqubits = nqubits
        self.q_depth = q_depth
        self.hidden_layer_pre = hidden_layer_pre if hidden_layer_pre is not None else []
        self.hidden_layer_post = hidden_layer_post if hidden_layer_post is not None else []
        self.activation = activation
        self._build_network()

    def _feature_map(self, qml_device):
        @qml.qnode(qml_device, interface="torch")
        def simple(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.nqubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(self.nqubits))
        
        @qml.qnode(qml_device, interface="torch")
        def product(inputs, weights):
            qml.AngleEmbedding(qml.math.asin(inputs), wires=range(self.nqubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(self.nqubits))
        
        @qml.qnode(qml_device, interface="torch")
        def chebyshev(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.nqubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(self.nqubits))
        
        return {"simple":simple, "product":product, "chebyshev":chebyshev}
    
    def _cost(self, qml_device):
        @qml.qnode(qml_device, interface="torch")
        def simple():
            return [qml.expval(qml.PauliZ(i)) for i in range(self.nqubits)]
        
        return {"simple":simple}

    def _ansatz(self, qml_device, iterations=1):
        @qml.qnode(qml_device, interface="torch")
        def simple(weights):
            qml.basic_entangler(weights, wires=range(self.nqubits))

        @qml.qnode(qml_device, interface="torch")
        def hea(weights):
            qml.strongly_entangling(weights, wires=range(self.nqubits))

        @qml.qnode(qml_device, interface="torch")
        def cascade(weights):
            qml.strongly_entangling(weights, wires=range(self.nqubits))

        return {"simple":simple, "hea":hea}

    def _qnn_layer(self, feature_map = "angle", ansatz = "hea", cost_func = "simple"):
        qml_device = qml.device("default.qubit", wires=self.nqubits)
        self._feature_map(qml_device)[feature_map]()
        self._ansatz(qml_device)[ansatz]()
        self._cost(qml_device)[cost_func]()

    def _build_network(self):
        layers = []

        # Pre-processing layers
        iter_layers = [self.input_num] + self.hidden_layer_pre + [self.nqubits]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        # Quantum layers
        q_layer = qml.qnn.TorchLayer(self._qnn_layer(), weight_shapes={"weights":(self.q_depth,self.nqubits)})
        layers.append(q_layer)
        layers.append(self.activation)
        
        # Post-processing layers
        iter_layers = [self.nqubits] + self.hidden_layer_post + [self.output_num]
        for i in range(len(iter_layers) - 1):
            layers.append(nn.Linear(iter_layers[i], iter_layers[i+1]))
            layers.append(self.activation)
        
        self.net = nn.Sequential(*layers)

class QCPINN(NN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        nqubits: Optional[int] = 4,
        hidden_layer_pre: Optional[List[int]] = None,
        hidden_layer_post: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__(input_vars, output_vars)
        self.nqubits = nqubits
        self.hidden_layer_pre = hidden_layer_pre if hidden_layer_pre is not None else []
        self.hidden_layer_post = hidden_layer_post if hidden_layer_post is not None else []
        self.activation = activation
        self._build_network()
    
    def _qnn_setup(self):

        qml_device = qml.device("default.qubit", wires=self.nqubits)

        @qml.qnode(qml_device, interface="torch")
        def _qnn_layer(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.nqubits), rotation="Y")

            # angle cascade ansatz
            for i in range(self.nqubits):
                qml.RX(weights[0, i], wires=i)
                qml.RY(weights[1, i], wires=i)
            for i in range(self.nqubits): # entanglement
                qml.CRX(weights[2, i], wires=[i, (i-1) % self.nqubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.nqubits)]

        
        qlayer = qml.qnn.TorchLayer(_qnn_layer, weight_shapes={"weights":(3, self.nqubits)})
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
