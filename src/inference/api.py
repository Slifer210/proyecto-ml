# src/inference/api.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from recommendation_pipeline import recommend_career

app = FastAPI(title="Career Recommendation API")

# Definir el input esperado
class UserInput(BaseModel):
    riasec: list  # [R, I, A, S, E, C]
    ocean: list   # 50 ítems IPIP

@app.get("/")
def root():
    # Redirigir automáticamente a Swagger
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(input: UserInput):
    # Solo pasamos riasec y ocean
    result = recommend_career(
        riasec_features=input.riasec,
        ocean_items=input.ocean,
        top_n=5
    )
    return result
