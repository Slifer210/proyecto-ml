from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from src.inference.recommendation_pipeline import recommend_career
from fastapi.responses import JSONResponse

# ==========================
# API FastAPI
# ==========================

app = FastAPI(
    title="Career Recommendation API",
    description="Servicio REST que combina RIASEC + OCEAN con afinidad (lazy loading)",
    version="1.0.0"
)

# Definir el input esperado
class UserInput(BaseModel):
    riasec: list  # [R, I, A, S, E, C]
    ocean: list   # 50 ítems del Big Five (IPIP)

@app.get("/")
def root():
    """Redirige automáticamente a la documentación interactiva (Swagger)."""
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(input: UserInput):
    """
    Recibe el perfil RIASEC y OCEAN y devuelve recomendaciones de carrera.
    Formato compatible con el cliente C# (ASP.NET).
    """
    try:
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=3
        )

        # Respuesta JSON con codificación explícita UTF-8
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
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            media_type="application/json; charset=utf-8"
        )

