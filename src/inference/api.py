from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from src.inference.recommendation_pipeline import recommend_career
import gc
import psutil

# ==========================
# API FastAPI
# ==========================

app = FastAPI(
    title="Career Recommendation API",
    description="Servicio REST que combina RIASEC + OCEAN con afinidad (lazy loading y control de memoria optimizado para Render)",
    version="1.0.1"
)


# ==========================
# Modelo de entrada
# ==========================

class UserInput(BaseModel):
    riasec: list  # 18 ítems del test RIASEC
    ocean: list   # 20 ítems del test OCEAN (Big Five)


# ==========================
# Función auxiliar: log de memoria
# ==========================

def log_memory_usage():
    """Muestra cuánta memoria usa el proceso (para debugging en Render)."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memoria usada: {mem_mb:.2f} MB")


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
    Recibe el perfil RIASEC y OCEAN y devuelve recomendaciones de carrera.
    Compatible con el cliente C# (ASP.NET) y pensado para Render Free Tier.
    """
    try:
        # Ejecutar pipeline híbrido
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=3
        )

        # Registrar uso de memoria
        log_memory_usage()

        # Liberar memoria después de cada request (importante en Render Free)
        gc.collect()

        # Enviar respuesta JSON (UTF-8 explícito)
        return JSONResponse(
            content={
                "riasec": result["riasec"],
                "subperfil": result["subperfil"],
                "ocean_vector": result["ocean_vector"],
                "recomendaciones": result["recomendaciones"]
            },
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        gc.collect()  # limpieza por si hubo excepción
        print(f"[ERROR] Falló /predict: {e}")

        return JSONResponse(
            content={"status": "error", "message": str(e)},
            media_type="application/json; charset=utf-8",
            status_code=500
        )
