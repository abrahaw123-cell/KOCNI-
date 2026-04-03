import numpy as np
from kocni_v8_quantum_engine import QuantumHolographicComputer

# =============================
# MOTOR BASE (V8)
# =============================

motor = QuantumHolographicComputer()


# =============================
# SISTEMA DE ANTICIPACIÓN
# =============================

class AnticipationModule:

    def __init__(self):

        self.ethics_threshold = 0.6
        self.stability_threshold = 0.5
        self.energy_threshold = 1.2


    def evaluate(self, result):

        ethics = result["ethics"]
        stability = result["stability"]
        energy = result["hamiltonian"]

        warnings = []

        if ethics < self.ethics_threshold:
            warnings.append("⚠️ riesgo ético")

        if stability < self.stability_threshold:
            warnings.append("⚠️ sistema inestable")

        if energy > self.energy_threshold:
            warnings.append("⚠️ energía anómala")

        return warnings


# =============================
# CORRECCIÓN DE DECISIÓN
# =============================

class DecisionCorrection:

    def __init__(self):

        self.anticipation = AnticipationModule()


    def process(self, text):

        result = motor.process(text)

        warnings = self.anticipation.evaluate(result)

        if len(warnings) > 0:

            decision = "REVISAR ACCIÓN"

        else:

            decision = "ACCIÓN SEGURA"

        return {

            "decision": decision,
            "warnings": warnings,
            "analysis": result

        }


# =============================
# EJECUCIÓN
# =============================

kocni = DecisionCorrection()


def run_kocni(text):

    result = kocni.process(text)

    print("\n===== KOCNI V8.1 =====")

    print("Decisión:", result["decision"])

    if result["warnings"]:
        print("Alertas:")
        for w in result["warnings"]:
            print("-", w)

    print("\nAnálisis interno:")

    print("Acción sugerida:", result["analysis"]["next_action"])

    print("Ética:", result["analysis"]["ethics"])

    print("Estabilidad:", result["analysis"]["stability"])

    print("Hamiltoniano:", result["analysis"]["hamiltonian"])

    print("Símbolos detectados:", result["analysis"]["top_symbols"])

    print("Moléculas:", result["analysis"]["molecules"])


# =============================
# PRUEBA
# =============================

if __name__ == "__main__":

    text = input("Introduce texto o código: ")

    run_kocni(text)
