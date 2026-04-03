import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import gradio as gr
from pyscf import gto, dft


# =============================
# 1. ANÁLISIS DE TEXTO
# =============================

def frequency_analysis(text):

    text = ''.join(c for c in text.upper() if c.isalpha())

    return Counter(text)


def text_to_signal(text, target_size=64):

    freq = frequency_analysis(text)

    values = np.array(list(freq.values()), dtype=float)

    signal = np.zeros(target_size)

    signal[:len(values)] = values

    signal = signal / (np.max(signal) + 1e-8)

    signal += 0.08 * np.sin(np.linspace(0, 20, target_size))

    return signal


# =============================
# 2. MAPEO SIMBÓLICO → MOLÉCULAS
# =============================

symbol_to_mol = {
    'A': 'H2O',
    'E': 'CH4',
    'O': 'NH3',
    'I': 'CO2',
    'P': 'C6H6',
    'R': 'DNA'
}


def compute_real_molecule_energy(mol_name):

    try:

        if mol_name == 'H2O':

            mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.92 0 0.24', basis='sto-3g')

        elif mol_name == 'CH4':

            mol = gto.M(atom='''
            C 0 0 0
            H 0.63 0.63 0.63
            H -0.63 -0.63 0.63
            H -0.63 0.63 -0.63
            H 0.63 -0.63 -0.63
            ''', basis='sto-3g')

        else:

            return 0.0

        mf = dft.RKS(mol)

        mf.xc = 'PBE'

        return mf.kernel()

    except:

        return -76.0


# =============================
# 3. MOTOR CUÁNTICO
# =============================

class QuantumHolographicComputer(nn.Module):

    def __init__(self, n_qubits=8):

        super().__init__()

        self.n_qubits = n_qubits

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(inputs, weights):

            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

        self.weights = np.random.random((6, n_qubits, 3))

        self.classical = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )


    # =============================
    # HOLOGRAFÍA
    # =============================

    def holographic_process(self, signal):

        boundary = signal[[0, 1, -2, -1]]

        alpha = np.linspace(0, 1, len(signal))

        recon = (1 - alpha) * boundary[0] + alpha * boundary[-1]

        recon += 0.25 * (
            (boundary[1] - boundary[0]) * (1 - alpha)
            - (boundary[-1] - boundary[-2]) * alpha
        )

        return torch.tensor(np.maximum(recon, 0), dtype=torch.float32)


    # =============================
    # PROCESAMIENTO PRINCIPAL
    # =============================

    def process(self, text):

        signal = text_to_signal(text)

        filtered = torch.tensor(signal, dtype=torch.float32)

        recon = self.holographic_process(filtered)

        q_out = self.circuit(recon[:self.n_qubits].numpy(), self.weights)

        q_tensor = torch.tensor(q_out, dtype=torch.float32).unsqueeze(0)

        output = self.classical(q_tensor)

        H = torch.norm(q_tensor)*0.65 + torch.mean(q_tensor**2)*1.35


        freq = frequency_analysis(text)

        top_sym = [s for s,_ in freq.most_common(4)]

        mols = [symbol_to_mol.get(s, 'Unknown') for s in top_sym]

        energies = [compute_real_molecule_energy(m) for m in mols]


        return {

            "next_action": output[0,0].item(),

            "ethics": torch.sigmoid(output[0,1]).item(),

            "stability": torch.sigmoid(output[0,2]).item(),

            "hamiltonian": H.item(),

            "top_symbols": top_sym,

            "molecules": mols,

            "energies": energies

        }


motor = QuantumHolographicComputer()


# =============================
# 4. EJECUCIÓN
# =============================

def run_quantum_computer(text):

    result = motor.process(text)

    mol_info = "\n".join(
        [f"{m}: {e:.4f} Hartree" for m, e in zip(result['molecules'], result['energies'])]
    )

    return (

        f"Próxima acción: {result['next_action']:.4f}\n"
        f"Ética: {result['ethics']:.3f}\n"
        f"Estabilidad: {result['stability']:.3f}\n"
        f"Hamiltoniano: {result['hamiltonian']:.4f}",

        f"Símbolos: {result['top_symbols']}\nMoléculas:\n{mol_info}"

    )


# =============================
# 5. VISUALIZACIÓN
# =============================

def create_visualizations(text):

    signal = text_to_signal(text)

    plt.figure(figsize=(10,6))

    plt.plot(signal)

    plt.title("Señal del Manuscrito")

    plt.savefig("quantum_viz.png")

    return "quantum_viz.png"


# =============================
# 6. INTERFAZ
# =============================

with gr.Blocks(title="KOCNI V8 Quantum Engine") as demo:

    gr.Markdown("# KOCNI V8 — Motor Cuántico Holográfico")

    text_input = gr.Textbox(label="Texto o código", lines=4)

    btn = gr.Button("Procesar")

    output1 = gr.Textbox(label="Estado del motor")

    output2 = gr.Textbox(label="Símbolos y moléculas")

    viz = gr.Image(label="Visualización")

    btn.click(run_quantum_computer, inputs=text_input, outputs=[output1, output2])

    btn.click(create_visualizations, inputs=text_input, outputs=viz)


# =============================
# 7. INICIO
# =============================

if __name__ == "__main__":

    print("Iniciando KOCNI V8...")

    demo.launch()
