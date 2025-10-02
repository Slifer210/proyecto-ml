import joblib
import pandas as pd
import json

# 1. Cargar modelos entrenados
model_riasec = joblib.load("../../models/riasec_model.pkl")
model_major = joblib.load("../../models/carrera_model.pkl")

# 2. Simular nueva respuesta
nueva_respuesta = {
    "R1": 3, "R2": 4, "R3": 2, "R4": 5, "R5": 3, "R6": 2, "R7": 4, "R8": 3,
    "I1": 5, "I2": 3, "I3": 4, "I4": 2, "I5": 5, "I6": 4, "I7": 2, "I8": 3,
    "A1": 2, "A2": 4, "A3": 3, "A4": 5, "A5": 3, "A6": 4, "A7": 2, "A8": 5,
    "S1": 3, "S2": 2, "S3": 4, "S4": 2, "S5": 5, "S6": 3, "S7": 2, "S8": 4,
    "E1": 4, "E2": 3, "E3": 4, "E4": 5, "E5": 3, "E6": 2, "E7": 3, "E8": 4,
    "C1": 3, "C2": 4, "C3": 2, "C4": 3, "C5": 4, "C6": 2, "C7": 5, "C8": 3
}
df_nuevo = pd.DataFrame([nueva_respuesta])

# ==============================
# 🔹 3. Predecir tipo RIASEC
# ==============================
probabilidades = model_riasec.predict_proba(df_nuevo)[0]
clases = model_riasec.classes_

ordenadas = sorted(zip(clases, probabilidades), key=lambda x: x[1], reverse=True)
top1 = ordenadas[0][0]
top2 = ordenadas[1][0]
combo = top1 + top2

# ==============================
# 🔹 4. Predecir Carrera (del dataset)
# ==============================
carrera_pred = model_major.predict(df_nuevo)[0]
probs_carreras = model_major.predict_proba(df_nuevo)[0]
ordenadas_carreras = sorted(
    zip(model_major.classes_, probs_carreras),
    key=lambda x: x[1],
    reverse=True
)

# ==============================
# 🔹 5. Preparar salida
# ==============================
salida = {
    "tipo_sugerido": top1,
    "top2": combo,
    "probabilidades_riasec": {c: round(p, 2) for c, p in ordenadas},
    "carrera_predicha_dataset": carrera_pred,
    "top3_carreras_dataset": [c for c, _ in ordenadas_carreras[:3]]
}

print(json.dumps(salida, indent=2, ensure_ascii=False))
