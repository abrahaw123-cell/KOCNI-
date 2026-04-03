import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from collections import Counter


# ============================
# CODIFICACIÓN DEL INPUT
# ============================

def encode_input(text, size=8):

    text = text.upper()

    freq = Counter(c for c in text if c.isalnum())

    values = np.array(list(freq.values()), dtype=float)

    signal = np.zeros(32)

    signal[:len(values)] = values[:32]

    signal = signal / (np.linalg.norm(signal) + 1e-8)

    idx = np.linspace(0, len(signal)-1, size)

    encoded = np.interp(idx, np.arange(len(signal)), signal)

    return torch.tensor(encoded, dtype=torch.float64)


# ============================
# CHIP CUÁNTICO
# ============================

class QuantumNodeChip(nn.Module):

    def __init__(self, n_qubits=8):

        super().__init__()

        self.n_qubits = n_qubits

        self.dev = qml.device("default.qubit", wires=n_qubits)

        self.weights = torch.randn(4, n_qubits, 3, dtype=torch.float64) * 0.1

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):

            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit


    # ============================
    # GEOMETRÍA CUÁNTICA
    # ============================

    def quantum_geometry(self, state):

        s = state.detach().numpy()

        metric = np.outer(s, s) + np.eye(len(s))*0.1

        eig = np.linalg.eigvalsh(metric)

        curvature = np.std(eig) / (np.mean(eig)+1e-8)

        volume = np.sqrt(abs(np.linalg.det(metric)))

        return {

            "curvature": float(curvature),

            "volume": float(volume)

        }


    # ============================
    # HAMILTONIANO
    # ============================

    def compute_hamiltonian(self, state):

        T = torch.norm(state)**2

        V = torch.sum(state**4)

        correlation = torch.sum(state*state)

        H = 0.5*T + 1.2*V - 0.3*correlation

        return float(H)


    # ============================
    # DETECCIÓN DE NODOS
    # ============================

    def detect_nodes(self, state):

        s = state.detach().numpy()

        threshold = np.mean(np.abs(s))

        nodes = []

        for i, amp in enumerate(s):

            if abs(amp) > threshold:

                nodes.append({

                    "id": i,

                    "amplitude": float(amp),

                    "energy": float(amp**2),

                    "type": "excited" if amp > 0 else "ground"

                })

        nodes.sort(key=lambda x: x["energy"], reverse=True)

        return nodes


    # ============================
    # PROCESO PRINCIPAL
    # ============================

    def process(self, text):

        encoded = encode_input(text, self.n_qubits)

        q_out = self.circuit(encoded, self.weights)

        state = torch.stack(q_out)

        geometry = self.quantum_geometry(state)

        H = self.compute_hamiltonian(state)

        nodes = self.detect_nodes(state)

        return {

            "quantum_state": state.detach().numpy(),

            "geometry": geometry,

            "hamiltonian": H,

            "nodes": nodes

        }


# ============================
# MÓDULO DE ANTICIPACIÓN
# ============================

class AnticipationSystem:

    def __init__(self):

        self.energy_threshold = 1.5


    def evaluate(self, result):

        warnings = []

        if result["hamiltonian"] > self.energy_threshold:

            warnings.append("alta energía detectada")

        if len(result["nodes"]) > 4:

            warnings.append("alta complejidad de nodos")

        return warnings


# ============================
# KOCNI V8.2
# ============================

class KOCNI:

    def __init__(self):

        self.chip = QuantumNodeChip()

        self.anticipation = AnticipationSystem()


    def analyze(self, text):

        result = self.chip.process(text)

        warnings = self.anticipation.evaluate(result)

        decision = "estable"

        if warnings:

            decision = "revisar"

        return {

            "decision": decision,

            "warnings": warnings,

            "analysis": result

        }


# ============================
# EJECUCIÓN
# ============================

if __name__ == "__main__":

    kocni = KOCNI()

    text = input("Introduce texto o ecuación: ")

    result = kocni.analyze(text)

    print("\n===== KOCNI V8.2 =====")

    print("Decisión:", result["decision"])

    print("Alertas:", result["warnings"])

    print("Hamiltoniano:", result["analysis"]["hamiltonian"])

    print("Curvatura:", result["analysis"]["geometry"]["curvature"])

    print("Nodos detectados:")

    for n in result["analysis"]["nodes"]:

        print(n)
