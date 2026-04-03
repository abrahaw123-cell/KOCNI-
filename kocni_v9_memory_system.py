import networkx as nx
import json
import numpy as np

# =========================
# MEMORIA COGNITIVA
# =========================

class CognitiveMemory:

    def __init__(self):

        self.graph = nx.Graph()

    def store_analysis(self, label, features):

        if label not in self.graph:
            self.graph.add_node(label, features=features)

        # conectar con nodos similares
        for node in self.graph.nodes:

            if node == label:
                continue

            other_features = self.graph.nodes[node]["features"]

            similarity = self.compute_similarity(features, other_features)

            if similarity > 0.8:

                self.graph.add_edge(label, node, weight=similarity)

    def compute_similarity(self, f1, f2):

        f1 = np.array(f1)
        f2 = np.array(f2)

        sim = np.dot(f1, f2) / (np.linalg.norm(f1)*np.linalg.norm(f2)+1e-8)

        return float(sim)

    def get_related(self, label):

        if label not in self.graph:
            return []

        neighbors = list(self.graph.neighbors(label))

        return neighbors


# =========================
# SISTEMA KOCNI V9
# =========================

class KOCNI:

    def __init__(self):

        self.memory = CognitiveMemory()

    def analyze(self, label, embedding):

        self.memory.store_analysis(label, embedding)

        related = self.memory.get_related(label)

        decision = "nuevo conocimiento"

        if related:
            decision = "patrón conocido"

        return {
            "decision": decision,
            "related_nodes": related
        }


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":

    kocni = KOCNI()

    emb1 = np.random.rand(16)
    emb2 = emb1 + np.random.normal(0,0.01,16)

    r1 = kocni.analyze("estructura_A", emb1)
    r2 = kocni.analyze("estructura_B", emb2)

    print("\n===== KOCNI V9 =====")

    print(r1)
    print(r2)
