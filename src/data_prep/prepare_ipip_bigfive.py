import pandas as pd

# ==========================
# 1. Cargar dataset
# ==========================
df = pd.read_csv("../../data/data-big-five.csv", sep="\t")

print("Columnas originales:", df.columns.tolist()[:20])  # muestra primeras

# ==========================
# 2. Definir ítems por factor
# ==========================
EXT = [f"EXT{i}" for i in range(1, 11)]
EST = [f"EST{i}" for i in range(1, 11)]
AGR = [f"AGR{i}" for i in range(1, 11)]
CSN = [f"CSN{i}" for i in range(1, 11)]
OPN = [f"OPN{i}" for i in range(1, 11)]

# Ítems que deben invertirse
reverse_items = [
    "EXT2","EXT4","EXT6","EXT8","EXT10",
    "EST2","EST4",
    "AGR1","AGR3","AGR5","AGR7",
    "CSN2","CSN4","CSN6","CSN8",
    "OPN2","OPN4","OPN6"
]

# ==========================
# 3. Invertir ítems reversos
# ==========================
for item in reverse_items:
    if item in df.columns:
        df[item] = 6 - df[item]

# ==========================
# 4. Calcular puntajes OCEAN
# ==========================
df["Extraversion"] = df[EXT].mean(axis=1)
df["Neuroticism"] = df[EST].mean(axis=1)   # EST mide neuroticismo directo
df["Agreeableness"] = df[AGR].mean(axis=1)
df["Conscientiousness"] = df[CSN].mean(axis=1)
df["Openness"] = df[OPN].mean(axis=1)

# ==========================
# 5. Guardar dataset limpio
# ==========================
bigfive = df[["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]]
bigfive.to_csv("../../data/bigfive_dataset_clean.csv", index=False)

print("Dataset OCEAN procesado y guardado en ../../data/bigfive_dataset_clean.csv")
