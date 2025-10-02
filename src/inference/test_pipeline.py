import joblib
import numpy as np
import json
from recommendation_pipeline import recommend_career

# ==========================
# 1. Cargar modelos y afinidad RIASEC
# ==========================
riasec_model = joblib.load("../../models/riasec_model.pkl")
career_model = joblib.load("../../models/career_model.pkl")
career_encoder = joblib.load("../../models/label_encoder.pkl")

with open("../../models/riasec_affinity.json", "r", encoding="utf-8") as f:
    riasec_affinity = json.load(f)

# ==========================
# 2. Casos de prueba (inputs simulados)
# ==========================
test_cases = {
    "R": {"riasec_input": [18, 5, 6, 4, 7, 5], "skill_input": [5, 4, 15, 8, 12, 6, 5, 7]},
    "I": {"riasec_input": [6, 18, 7, 5, 6, 4], "skill_input": [10, 5, 7, 18, 14, 8, 6, 10]},
    "A": {"riasec_input": [4, 6, 18, 7, 8, 5], "skill_input": [12, 16, 8, 10, 9, 7, 6, 8]},
    "S": {"riasec_input": [5, 6, 8, 18, 10, 7], "skill_input": [14, 6, 7, 12, 9, 17, 16, 8]},
    "E": {"riasec_input": [6, 7, 8, 10, 18, 9], "skill_input": [11, 5, 7, 12, 14, 10, 9, 8]},
    "C": {"riasec_input": [7, 6, 5, 8, 9, 18], "skill_input": [8, 4, 6, 11, 9, 7, 10, 15]},
}

# ==========================
# 3. Ejecutar pruebas híbridas
# ==========================
if __name__ == "__main__":
    for profile, data in test_cases.items():
        result = recommend_career(
            data["riasec_input"],
            data["skill_input"],
            top_n=10,
            penalty=0.7,   # penalizamos carreras no afines
        )

        riasec_label = result["riasec"]
        ajustado = result["ajustado"]
        filtrado = result["filtrado"]

        print("\n=== Perfil de prueba:", profile, "===")
        print("Perfil RIASEC predicho:", riasec_label)

        print("\nTop 5 Ajustado (boost/penalty):")
        for career, score in ajustado[:5]:
            print(f"- {career} ({score:.2f})")

        print("\nTop 5 Filtrado (afinidad fuzzy O*NET):")
        if filtrado:
            for career, score in filtrado[:5]:
                print(f"- {career} ({score:.2f})")
        else:
            print("No se encontraron carreras en la afinidad para este perfil.")
