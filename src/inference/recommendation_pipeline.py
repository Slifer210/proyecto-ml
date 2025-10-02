import joblib
import numpy as np
import pandas as pd
import json
from fuzzywuzzy import process
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================
# 1. Cargar modelos y afinidad
# ==========================
riasec_model = joblib.load("../../models/riasec_model.pkl")
ocean_model = joblib.load("../../models/ocean_model.pkl")

with open("../../models/riasec_affinity.json", "r", encoding="utf-8") as f:
    riasec_affinity = json.load(f)

# ==========================
# Funciones auxiliares
# ==========================
def normalize(name: str) -> str:
    """Normaliza nombres de carreras para comparación fuzzy."""
    return name.lower().split("(")[0].strip()

def fuzzy_match(career_str, riasec_label, score_cutoff=75):
    """Verifica si una carrera tiene afinidad con el perfil RIASEC."""
    target_list = riasec_affinity.get(riasec_label, [])
    if not target_list:
        return False
    target_list_norm = [normalize(t) for t in target_list]
    career_norm = normalize(career_str)
    best_match, best_score = process.extractOne(career_norm, target_list_norm)
    return best_score >= score_cutoff

# ==========================
# 2. Pipeline híbrido
# ==========================
def recommend_career(
    riasec_features,
    ocean_items,
    top_n=5,
    weight_riasec=1.2,
    weight_ocean=0.2
):
    """Pipeline RIASEC + OCEAN con diccionario de afinidad."""

    # --- Paso 1: predecir perfil RIASEC ---
    riasec_input = pd.DataFrame([riasec_features],
                                columns=["R", "I", "A", "S", "E", "C"])
    riasec_pred = riasec_model.predict(riasec_input)[0]
    riasec_label = str(riasec_pred)

    # --- Paso 2: predecir OCEAN ---
    item_cols = list(ocean_model.estimators_[0].feature_names_in_)
    ocean_input = pd.DataFrame([ocean_items], columns=item_cols)
    ocean_vector = ocean_model.predict(ocean_input)[0]  # [O, C, E, A, N]

    # --- Paso 3: recomendar desde el diccionario ---
    carreras = riasec_affinity.get(riasec_label, [])
    adjusted = []
    for career in carreras:
        score = 1.0  # base
        # Boost con RIASEC
        score *= weight_riasec
        # Ajuste con OCEAN (ejemplo simple usando Openness + Conscientiousness)
        ocean_boost = (ocean_vector[0] + ocean_vector[1]) / 2
        score *= (1 + weight_ocean * ocean_boost)
        adjusted.append((career, score))

    adjusted_final = sorted(adjusted, key=lambda x: x[1], reverse=True)[:top_n]

    return {
        "riasec": riasec_label,
        "ocean_vector": ocean_vector.tolist(),
        "recomendaciones": adjusted_final
    }

# ==========================
# 3. API con FastAPI
# ==========================
app = FastAPI(
    title="Career Recommendation API",
    description="Servicio REST que combina RIASEC + OCEAN con un diccionario de afinidad",
    version="1.0.0"
)

class UserInput(BaseModel):
    riasec: List[int]   # [R, I, A, S, E, C]
    ocean: List[int]    # 50 ítems del Big Five (IPIP)

@app.get("/")
def home():
    return {"docs": "Visita /docs para probar la API con Swagger"}

@app.post("/predict")
def predict(input: UserInput):
    try:
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=5
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
