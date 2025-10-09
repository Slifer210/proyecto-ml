from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from src.inference.recommendation_pipeline import recommend_career
import gc
import psutil
import os
import gdown

# ==========================
# API FastAPI
# ==========================

app = FastAPI(
    title="Career Recommendation API",
    description=(
        "Servicio REST que combina RIASEC + OCEAN con afinidad. "
        "Optimizado para Render Free Tier (512 MB, carga diferida y control de memoria)."
    ),
    version="1.2.0"
)

# ==========================
# Modelo de entrada
# ==========================

class UserInput(BaseModel):
    riasec: list  # 6, 18 o más ítems del test RIASEC
    ocean: list   # 20 ítems del test OCEAN (Big Five)

# ==========================
# Funciones auxiliares
# ==========================

def log_memory_usage():
    """Muestra cuánta memoria usa el proceso (para debugging en Render)."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memoria usada: {mem_mb:.2f} MB")


def ensure_models():
    """Descarga los modelos una sola vez al iniciar el contenedor, si no existen."""
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(MODELS_DIR, exist_ok=True)

    FILES = {
        "riasec_model.pkl": os.getenv("RIASEC_URL"),
        "ocean_model.pkl": os.getenv("OCEAN_URL"),
        "riasec_affinity.json": os.getenv("AFFINITY_URL"),
    }

    for fname, url in FILES.items():
        dest = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(dest) and url:
            try:
                print(f"Descargando {fname} desde {url} ...")
                gdown.download(url, dest, quiet=False)
                print(f"{fname} listo en {dest}")
            except Exception as e:
                print(f"No se pudo descargar {fname}: {e}")
        else:
            print(f"✔️ {fname} ya existe, no se descarga nuevamente.")


# ==========================
# Eventos de inicio
# ==========================

@app.on_event("startup")
def startup_event():
    """Se ejecuta al iniciar el contenedor en Render."""
    print("Iniciando servidor FastAPI y verificando modelos...")
    ensure_models()
    log_memory_usage()
    print("Modelos verificados. Servidor listo para recibir peticiones.")


# ==========================
# Endpoints principales
# ==========================

@app.get("/")
def root():
    """Redirige automáticamente a la documentación interactiva (Swagger)."""
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(input: UserInput):
    """
    Recibe los puntajes RIASEC y OCEAN y devuelve recomendaciones de carrera.
    Optimizado para Render Free Tier (512 MB).
    """
    try:
        # Ejecutar pipeline híbrido secuencial
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=3
        )

        # Log de uso de memoria
        log_memory_usage()

        # Limpieza de memoria tras cada request
        gc.collect()

        # Respuesta JSON
        return JSONResponse(
            content={
                "status": "ok",
                "result": {
                    "riasec": result["riasec"],
                    "subperfil": result["subperfil"],
                    "ocean_vector": result["ocean_vector"],
                    "recomendaciones": result["recomendaciones"]
                }
            },
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        gc.collect()  # limpieza en caso de excepción
        print(f"[ERROR] Falló /predict: {e}")

        return JSONResponse(
            content={"status": "error", "message": str(e)},
            media_type="application/json; charset=utf-8",
            status_code=500
        )
