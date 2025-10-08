import os
import joblib
import pandas as pd
import json
import gdown
from deep_translator import GoogleTranslator
from fuzzywuzzy import process
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse

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
        print(f"Descargando {fname} desde {url} ...")
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
        print("Cargando modelo RIASEC...")
        _riasec_model = joblib.load(path)
    return _riasec_model

def get_ocean_model():
    global _ocean_model
    if _ocean_model is None:
        path = os.path.join(MODELS_DIR, "ocean_model.pkl")
        print("Cargando modelo OCEAN...")
        _ocean_model = joblib.load(path)
    return _ocean_model

def get_affinity():
    global _riasec_affinity
    if _riasec_affinity is None:
        path = os.path.join(MODELS_DIR, "riasec_affinity.json")
        print("Cargando diccionario de afinidad...")
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

def traducir_lista(carreras):
    """Traduce dinámicamente una lista de carreras del inglés al español (latino)."""
    try:
        traductor = GoogleTranslator(source='en', target='es')
        return [traductor.translate(c) for c in carreras]
    except Exception as e:
        print(f"Error al traducir: {e}")
        return carreras  

# ==========================
# 3. Pipeline híbrido RIASEC + OCEAN
# ==========================

def get_subprofile(riasec_vector):
    """
    Determina el subperfil RIASEC-Perú (R-Tech, I-Science, A-Diseño, etc.)
    según las dos letras con mayor puntaje del vector RIASEC.
    """
    subprofiles = {
        "R": {"C": "R-Tech", "E": "R-Ind", "A": "R-Build", "S": "R-Agro", "I": "R-Geo"},
        "I": {"R": "I-Tech", "A": "I-Science", "S": "I-Health", "C": "I-Analytic", "E": "I-Economic"},
        "A": {"S": "A-ComunicaciónVisual", "E": "A-Diseño", "I": "A-ArtesEscénicas", "R": "A-ArtesPlásticas", "C": "A-PatrimonioCultural"},
        "S": {"E": "S-Comunitario", "I": "S-Psicológico", "A": "S-Educativo", "C": "S-Salud", "R": "S-DeporteYRecreación"},
        "E": {"C": "E-Negocios", "A": "E-MarketingYComercio", "S": "E-DerechoYGestiónPública", "I": "E-EmpresarialTecnológico", "R": "E-EmpresarialIndustrial"},
        "C": {"R": "C-Informático", "E": "C-ContableFinanciero", "S": "C-Administrativo", "I": "C-EstadísticoAnalítico", "A": "C-Ofimático"}
    }

    letters = ["R", "I", "A", "S", "E", "C"]
    scores = pd.Series(riasec_vector, index=letters)
    top_two = scores.nlargest(2).index.tolist()
    first, second = top_two
    subperfil = subprofiles.get(first, {}).get(second)
    if not subperfil:
        subperfil = subprofiles.get(second, {}).get(first)
    return subperfil if subperfil else first


def recommend_career(
    riasec_features,
    ocean_items,
    top_n=3,
    weight_riasec=1.2,
    weight_ocean=0.2
):
    """Pipeline híbrido RIASEC + OCEAN con soporte para subperfiles peruanos."""

    # --- Paso 1: cargar modelos y afinidad ---
    riasec_model = get_riasec_model()
    ocean_model = get_ocean_model()
    riasec_affinity = get_affinity()

    # --- Paso 2: predecir perfil principal RIASEC ---
    riasec_input = pd.DataFrame([riasec_features], columns=["R", "I", "A", "S", "E", "C"])
    riasec_pred = riasec_model.predict(riasec_input)[0]
    riasec_label = str(riasec_pred)

    # --- Paso 3: determinar subperfil ---
    sub_label = get_subprofile(riasec_features)

    # --- Paso 4: mapear subperfil a JSON ---
    alias_map = {
        # R
        "R-Build": "R-Build", "R-Tech": "R-Tech", "R-Ind": "R-Ind", "R-Geo": "R-Geo",
        "R-Agro": "R-Agro", "R-HealthTech": "R-HealthTech", "R-Energy": "R-Energy",
        # I
        "I-Science": "I-Científico", "I-Health": "I-Médico", "I-Analytic": "I-Analítico",
        "I-Tech": "I-Tecnológico", "I-Social": "I-SocialCientífico", "I-Geo": "I-AstroGeo",
        # A
        "A-Design": "A-Diseño", "A-Visual": "A-ComunicaciónVisual", "A-Performing": "A-ArtesEscénicas",
        "A-Music": "A-Música", "A-Plastic": "A-ArtesPlásticas", "A-Culture": "A-PatrimonioCultural",
        # S
        "S-Edu": "S-Educativo", "S-Health": "S-Salud", "S-Psy": "S-Psicológico",
        "S-Community": "S-Comunitario", "S-Mental": "S-SaludMental", "S-Sport": "S-DeporteYRecreación",
        # E
        "E-Business": "E-Negocios", "E-Finance": "E-EconomíaFinanzas", "E-Communication": "E-ComunicaciónLiderazgo",
        "E-Marketing": "E-MarketingYComercio", "E-Law": "E-DerechoYGestiónPública", "E-TechBusiness": "E-EmpresarialTecnológico",
        # C
        "C-Admin": "C-Administrativo", "C-Finance": "C-ContableFinanciero", "C-IT": "C-Informático",
        "C-Logistic": "C-Logístico", "C-Statistic": "C-EstadísticoAnalítico", "C-Office": "C-Ofimático"
    }

    mapped_label = alias_map.get(sub_label, sub_label)

    # --- Paso 5: extraer carreras del subperfil ---
    carreras_data = []
    if "-" in mapped_label:
        base, sub = mapped_label.split("-", 1)
        carreras_data = riasec_affinity.get(base, {}).get(mapped_label, [])
    else:
        sub_aff = riasec_affinity.get(riasec_label, {})
        for subblock in sub_aff.values():
            carreras_data.extend(subblock)

    # --- Paso 6: predecir vector OCEAN ---
    item_cols = list(ocean_model.estimators_[0].feature_names_in_)
    ocean_input = pd.DataFrame([ocean_items], columns=item_cols)
    ocean_vector = ocean_model.predict(ocean_input)[0]

    # --- Paso 7: calcular afinidad ---
    adjusted = []
    for entry in carreras_data:
        carrera = entry["carrera"]
        universidades = entry.get("universidades", [])
        score = 1.0 * weight_riasec
        ocean_boost = (ocean_vector[0] + ocean_vector[1]) / 2
        score *= (1 + weight_ocean * ocean_boost)
        adjusted.append({
            "carrera": carrera,
            "universidades": universidades,
            "score": round(score, 3)
        })

    adjusted_final = sorted(adjusted, key=lambda x: x["score"], reverse=True)[:top_n]

    return {
        "riasec": riasec_label,
        "subperfil": sub_label,
        "ocean_vector": ocean_vector.tolist(),
        "recomendaciones": adjusted_final
    }

# ==========================
# 4. API con FastAPI
# ==========================

app = FastAPI(
    title="API de Recomendación Vocacional",
    description="Servicio REST que combina RIASEC + OCEAN con afinidad y traducción automática",
    version="1.1.0"
)

class UserInput(BaseModel):
    riasec: List[int]
    ocean: List[int]

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API de orientación vocacional. Visita /docs para probarla."}

@app.post("/predict")
def predict(input: UserInput):
    try:
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=3
        )
        return JSONResponse(
            content={"status": "ok", "result": result},
            ensure_ascii=False
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}
