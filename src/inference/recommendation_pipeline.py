import os
import joblib
import pandas as pd
import json
import gdown
from fuzzywuzzy import process
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================
# 1. Descarga modelos si no existen
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
)

os.makedirs(MODELS_DIR, exist_ok=True)

# URLs desde variables de entorno (Render → Environment Variables)
FILES = {
    "riasec_model.pkl": os.getenv("RIASEC_URL"),
    "ocean_model.pkl": os.getenv("OCEAN_URL"),
    "riasec_affinity.json": os.getenv("AFFINITY_URL"),
}

for fname, url in FILES.items():
    dest = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(dest) and url:
        print(f"📥 Descargando {fname} desde {url} ...")
        gdown.download(url, dest, quiet=False)

# ==========================
# 2. Lazy loading (solo carga cuando se necesita)
# ==========================

_riasec_model = None
_ocean_model = None
_riasec_affinity = None

def get_riasec_model():
    global _riasec_model
    if _riasec_model is None:
        path = os.path.join(MODELS_DIR, "riasec_model.pkl")
        print("🔄 Cargando modelo RIASEC...")
        _riasec_model = joblib.load(path)
    return _riasec_model

def get_ocean_model():
    global _ocean_model
    if _ocean_model is None:
        path = os.path.join(MODELS_DIR, "ocean_model.pkl")
        print("🔄 Cargando modelo OCEAN...")
        _ocean_model = joblib.load(path)
    return _ocean_model

def get_affinity():
    global _riasec_affinity
    if _riasec_affinity is None:
        path = os.path.join(MODELS_DIR, "riasec_affinity.json")
        print("🔄 Cargando diccionario de afinidad...")
        with open(path, "r", encoding="utf-8") as f:
            _riasec_affinity = json.load(f)
    return _riasec_affinity

# ==========================
# Funciones auxiliares
# ==========================

def normalize(name: str) -> str:
    """Normaliza nombres de carreras para comparación fuzzy."""
    return name.lower().split("(")[0].strip()

def fuzzy_match(career_str, riasec_label, score_cutoff=75):
    """Verifica si una carrera tiene afinidad con el perfil RIASEC."""
    riasec_affinity = get_affinity()
    target_list = riasec_affinity.get(riasec_label, [])
    if not target_list:
        return False
    target_list_norm = [normalize(t) for t in target_list]
    career_norm = normalize(career_str)
    best_match, best_score = process.extractOne(career_norm, target_list_norm)
    return best_score >= score_cutoff

# ==========================
# 3. Pipeline híbrido
# ==========================

def recommend_career(
    riasec_features,
    ocean_items,
    top_n=5,
    weight_riasec=1.2,
    weight_ocean=0.2
):
    """Pipeline RIASEC + OCEAN con diccionario de afinidad."""

    riasec_model = get_riasec_model()
    ocean_model = get_ocean_model()
    riasec_affinity = get_affinity()

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
        score *= weight_riasec
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
# 4. API con FastAPI
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
