import os
import gc
import joblib
import pandas as pd
import json
import psutil
import gdown
from fuzzywuzzy import process

# ==========================
# 1. Configuración de rutas y descarga de modelos
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
# 2. Funciones auxiliares
# ==========================

def normalize(name: str) -> str:
    """Normaliza nombres de carreras para comparación fuzzy."""
    return name.lower().split("(")[0].strip()


def fuzzy_match(career_str, riasec_label, score_cutoff=75):
    """Verifica si una carrera tiene afinidad con el perfil RIASEC."""
    with open(os.path.join(MODELS_DIR, "riasec_affinity.json"), "r", encoding="utf-8") as f:
        riasec_affinity = json.load(f)
    target_list = riasec_affinity.get(riasec_label, [])
    if not target_list:
        return False
    target_list_norm = [normalize(t) for t in target_list]
    career_norm = normalize(career_str)
    best_match, best_score = process.extractOne(career_norm, target_list_norm)
    return best_score >= score_cutoff


def log_memory():
    """Registra en consola el consumo de memoria actual (MB)."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memoria usada: {mem:.2f} MB")


# ==========================
# 3. Determinar subperfil RIASEC
# ==========================

def get_subprofile(riasec_vector):
    """Determina el subperfil RIASEC-Perú según las dos letras dominantes."""
    subprofiles = {
        "R": {"C": "R-Tech", "E": "R-Ind", "A": "R-Build", "S": "R-Agro", "I": "R-Geo"},
        "I": {"R": "I-Tech", "A": "I-Science", "S": "I-Health", "C": "I-Analytic", "E": "I-Economic"},
        "A": {"S": "A-ComunicaciónVisual", "E": "A-Diseño", "I": "A-ArtesEscénicas",
              "R": "A-ArtesPlásticas", "C": "A-PatrimonioCultural"},
        "S": {"E": "S-Comunitario", "I": "S-Psicológico", "A": "S-Educativo",
              "C": "S-Salud", "R": "S-DeporteYRecreación"},
        "E": {"C": "E-Negocios", "A": "E-MarketingYComercio", "S": "E-DerechoYGestiónPública",
              "I": "E-EmpresarialTecnológico", "R": "E-EmpresarialIndustrial"},
        "C": {"R": "C-Informático", "E": "C-ContableFinanciero", "S": "C-Administrativo",
              "I": "C-EstadísticoAnalítico", "A": "C-Ofimático"},
    }

    letters = ["R", "I", "A", "S", "E", "C"]
    scores = pd.Series(riasec_vector, index=letters)
    top_two = scores.nlargest(2).index.tolist()
    first, second = top_two
    subperfil = subprofiles.get(first, {}).get(second)
    if not subperfil:
        subperfil = subprofiles.get(second, {}).get(first)
    return subperfil if subperfil else first


# ==========================
# 4. Pipeline principal (optimizando RAM y entrada flexible)
# ==========================

def recommend_career(
    riasec_features,
    ocean_items,
    top_n=3,
    weight_riasec=1.2,
    weight_ocean=0.2
):
    """
    Pipeline híbrido RIASEC + OCEAN optimizado para Render Free Tier (512 MB).
    - Acepta 6, 18 o más ítems RIASEC.
    - Carga modelos de forma secuencial (uno a la vez).
    """

    try:
        # --- Paso 0: Ajustar entrada RIASEC (6, 18 o más ítems) ---
        if len(riasec_features) > 6:
            n = len(riasec_features)
            # Divide en 6 bloques proporcionales
            group_size = n // 6
            grouped = [
                sum(riasec_features[i:i+group_size]) / group_size
                for i in range(0, n, group_size)
            ]
            grouped = grouped[:6]  # garantiza que haya solo 6 valores
            print(f"[INFO] RIASEC agrupado automáticamente ({n} → 6)")
        else:
            grouped = riasec_features

        # --- Paso 1: cargar afinidad ---
        with open(os.path.join(MODELS_DIR, "riasec_affinity.json"), "r", encoding="utf-8") as f:
            riasec_affinity = json.load(f)

        # --- Paso 2: cargar y usar modelo RIASEC ---
        print("Cargando modelo RIASEC (temporal)...")
        riasec_model = joblib.load(os.path.join(MODELS_DIR, "riasec_model.pkl"))

        riasec_input = pd.DataFrame([grouped], columns=["R", "I", "A", "S", "E", "C"])
        riasec_pred = riasec_model.predict(riasec_input)[0]
        riasec_label = str(riasec_pred)
        sub_label = get_subprofile(grouped)

        del riasec_model, riasec_input
        gc.collect()

        # --- Paso 3: mapear etiquetas ---
        alias_map = {
            "R-Tech": "R-Tech", "R-Ind": "R-Ind", "R-Build": "R-Build", "R-Geo": "R-Geo", "R-Agro": "R-Agro",
            "I-Science": "I-Científico", "I-Health": "I-Médico", "I-Analytic": "I-Analítico", "I-Tech": "I-Tecnológico",
            "A-Diseño": "A-Diseño", "A-ComunicaciónVisual": "A-ComunicaciónVisual", "A-ArtesEscénicas": "A-ArtesEscénicas",
            "S-Comunitario": "S-Comunitario", "S-Educativo": "S-Educativo", "S-Salud": "S-Salud",
            "E-Negocios": "E-Negocios", "E-MarketingYComercio": "E-MarketingYComercio",
            "C-Informático": "C-Informático", "C-ContableFinanciero": "C-ContableFinanciero",
        }
        mapped_label = alias_map.get(sub_label, sub_label)

        # --- Paso 4: extraer carreras ---
        carreras_data = []
        if "-" in mapped_label:
            base, sub = mapped_label.split("-", 1)
            carreras_data = riasec_affinity.get(base, {}).get(mapped_label, [])
        else:
            sub_aff = riasec_affinity.get(riasec_label, {})
            for subblock in sub_aff.values():
                carreras_data.extend(subblock)

        del riasec_affinity
        gc.collect()

        # --- Paso 5: cargar y usar modelo OCEAN ---
        print("Cargando modelo OCEAN (temporal)...")
        ocean_model = joblib.load(os.path.join(MODELS_DIR, "ocean_model.pkl"))

        item_cols = list(ocean_model.estimators_[0].feature_names_in_)
        ocean_input = pd.DataFrame([ocean_items], columns=item_cols)
        ocean_vector = ocean_model.predict(ocean_input)[0]

        del ocean_model, ocean_input
        gc.collect()

        # --- Paso 6: combinar afinidades ---
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

        del carreras_data, ocean_vector
        gc.collect()

        adjusted_final = sorted(adjusted, key=lambda x: x["score"], reverse=True)[:top_n]

        # --- Paso 7: log de memoria ---
        log_memory()
        gc.collect()

        return {
            "riasec": riasec_label,
            "subperfil": sub_label,
            "recomendaciones": adjusted_final
        }

    except Exception as e:
        gc.collect()
        print(f"[ERROR] recommend_career(): {e}")
        raise e
