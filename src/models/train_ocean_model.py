# src/models/train_ocean_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ==========================
# 1. Cargar dataset
# ==========================
# Dataset crudo con ítems
df_items = pd.read_csv("../../data/data-big-five.csv", sep="\t")

# Dataset limpio con puntajes OCEAN
df_ocean = pd.read_csv("../../data/bigfive_dataset_clean.csv")

print("Shapes originales:", df_items.shape, df_ocean.shape)

# ==========================
# 2. Selección de features
# ==========================
# 🔹 Usamos solo los 20 ítems seleccionados
item_cols = [
    # Extraversion
    "EXT1", "EXT2", "EXT3", "EXT4",
    # Amabilidad
    "AGR1", "AGR2", "AGR3", "AGR4",
    # Responsabilidad (Conscientiousness)
    "CSN1", "CSN2", "CSN3", "CSN4",
    # Estabilidad emocional (Neuroticism inverso)
    "EST1", "EST2", "EST3", "EST4",
    # Apertura
    "OPN1", "OPN2", "OPN3", "OPN4"
]

# Features (X) y Target (y)
X = df_items[item_cols]
y = df_ocean[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]]

# ==========================
# 3. Limpiar y muestrear
# ==========================
df_full = pd.concat([X, y], axis=1)
print("Antes de dropna:", df_full.shape)

df_full = df_full.dropna()
print("Después de dropna:", df_full.shape)

# 🔹 Opcional: muestreo para entrenar más rápido
df_full = df_full.sample(n=50000, random_state=42)
print("Después de muestreo:", df_full.shape)

X = df_full[item_cols]
y = df_full[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]]

# ==========================
# 4. Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================
# 5. Entrenar modelo
# ==========================
model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
))

print("Entrenando modelo OCEAN con 20 ítems...")
model.fit(X_train, y_train)

# ==========================
# 6. Evaluación
# ==========================
y_pred = model.predict(X_test)

print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R2 Score (por dimensión):", r2_score(y_test, y_pred, multioutput="raw_values"))
print("R2 Promedio:", r2_score(y_test, y_pred, multioutput="uniform_average"))

# ==========================
# 7. Guardar modelo
# ==========================
os.makedirs("../../models", exist_ok=True)
joblib.dump(model, "../../models/ocean_model.pkl")

print("\nModelo OCEAN guardado en ../../models/ocean_model.pkl")
