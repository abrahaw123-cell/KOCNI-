import numpy as np
import torch
import torch.nn as nn
from scipy.special import sph_harm


# ==========================
# EXPANSIÓN EN ARMÓNICOS
# ==========================

def spherical_embedding(points, l_max=3):

    embeddings = []

    for l in range(l_max + 1):

        for m in range(-l, l + 1):

            values = []

            for x, y, z in points:

                r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)

                Y = sph_harm(m, l, phi, theta)

                values.append(np.real(Y))

            embeddings.append(np.mean(values))

    return np.array(embeddings)


# ==========================
# DENSIDAD ELECTRÓNICA SIMPLE
# ==========================

def approximate_density(points):

    density = []

    for x, y, z in points:

        r = np.sqrt(x**2 + y**2 + z**2)

        rho = np.exp(-r**2)

        density.append(rho)

    return np.array(density)


# ==========================
# MODELO GEOMÉTRICO
# ==========================

class QuantumGeometryModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):

        return self.net(x)


# ==========================
# ANALIZADOR GEOMÉTRICO
# ==========================

class QuantumGeometryAnalyzer:

    def __init__(self):

        self.model = QuantumGeometryModel()

    def analyze_structure(self, points):

        rho = approximate_density(points)

        embed = spherical_embedding(points)

        features = np.concatenate([embed[:8], rho[:8]])

        tensor = torch.tensor(features, dtype=torch.float32)

        out = self.model(tensor)

        return {
            "energy_estimate": float(out[0]),
            "stability": float(torch.sigmoid(out[1])),
            "reactivity": float(torch.sigmoid(out[2])),
            "embedding": embed.tolist()
        }


# ==========================
# KOCNI V8.3
# ==========================

class KOCNI:

    def __init__(self):

        self.geometry = QuantumGeometryAnalyzer()

    def analyze_points(self, points):

        result = self.geometry.analyze_structure(points)

        decision = "estable"

        if result["reactivity"] > 0.6:

            decision = "alta reactividad"

        return {
            "decision": decision,
            "analysis": result
        }


# ==========================
# EJECUCIÓN
# ==========================

if __name__ == "__main__":

    kocni = KOCNI()

    # ejemplo de estructura
    points = np.random.randn(10,3)

    result = kocni.analyze_points(points)

    print("\n===== KOCNI V8.3 =====")
    print("Decisión:", result["decision"])
    print("Energía estimada:", result["analysis"]["energy_estimate"])
    print("Estabilidad:", result["analysis"]["stability"])
    print("Reactividad:", result["analysis"]["reactivity"])
