from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from src.inference.recommendation_pipeline import recommend_career

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
    """Recibe el perfil RIASEC y OCEAN y devuelve recomendaciones de carrera."""
    try:
        result = recommend_career(
            riasec_features=input.riasec,
            ocean_items=input.ocean,
            top_n=5
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
