import pandas as pd
import json

# ==========================
# 1. Cargar archivo
# ==========================
file_path = "../../data/Interests.xlsx"
df = pd.read_excel(file_path)

print("Columnas originales:", df.columns.tolist())
print("Ejemplo de datos:\n", df.head())

# ==========================
# 2. Pivotear la tabla
# ==========================
pivot = df.pivot_table(index="Title",
                       columns="Element Name",
                       values="Data Value",
                       aggfunc="mean").reset_index()

# Renombrar columnas a siglas RIASEC
rename_map = {
    "Realistic": "R",
    "Investigative": "I",
    "Artistic": "A",
    "Social": "S",
    "Enterprising": "E",
    "Conventional": "C"
}
pivot = pivot.rename(columns=rename_map)

print("\nEjemplo pivotado:\n", pivot.head())

# ==========================
# 3. Construcción diccionario
# ==========================
riasec_columns = ["R", "I", "A", "S", "E", "C"]
riasec_affinity = {k: [] for k in riasec_columns}

for _, row in pivot.iterrows():
    dominant = row[riasec_columns].idxmax()   # letra dominante
    occupation = str(row["Title"]).strip()
    if occupation not in riasec_affinity[dominant]:
        riasec_affinity[dominant].append(occupation)

# ==========================
# 4. Guardar como JSON
# ==========================
with open("../../models/riasec_affinity.json", "w", encoding="utf-8") as f:
    json.dump(riasec_affinity, f, indent=4, ensure_ascii=False)

print("\nDiccionario guardado en ../../models/riasec_affinity.json")
