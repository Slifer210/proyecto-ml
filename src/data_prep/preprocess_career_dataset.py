import pandas as pd

# ==========================
# 1. Cargar dataset bruto
# ==========================
file_path = "../../data/career_dataset.csv"

try:
    # Detecta automáticamente el delimitador y maneja comillas
    df = pd.read_csv(file_path, encoding="utf-8-sig", sep=None, engine="python", quotechar='"')
except Exception as e:
    print(" Error al leer con sep=None, probando con tabulador...")
    df = pd.read_csv(file_path, encoding="utf-8-sig", sep="\t", engine="python", quotechar='"')

print("Dataset original:", df.shape)
print("Columnas detectadas:", df.columns.tolist())

# ==========================
# 2. Normalizar columnas
# ==========================
df.columns = (
    df.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace("-", "_")
              .str.replace(".", "")
)

# Ajustar el nombre si hay variantes
if "Logical___Mathematical" in df.columns:
    df = df.rename(columns={"Logical___Mathematical": "Logical_Mathematical"})

# ==========================
# 3. Eliminar columnas irrelevantes
# ==========================
cols_to_drop = [c for c in df.columns if c.startswith("P") or c in ["SrNo", "SrNo_", "Course", "Student", "s/p"]]
df = df.drop(columns=cols_to_drop, errors="ignore")

# ==========================
# 4. Limpiar columna de profesión
# ==========================
if "Job_profession" in df.columns:
    df["Job_profession"] = (
        df["Job_profession"].astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
else:
    raise KeyError("No se encontró la columna 'Job_profession' en el dataset.")

# ==========================
# 5. Convertir valores a numérico
# ==========================
for col in df.columns:
    if col != "Job_profession":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================
# 6. Eliminar filas vacías
# ==========================
df = df.dropna(how="all")
df = df.dropna(subset=["Job_profession"])

# ==========================
# 7. Reporte de nulos
# ==========================
print("\nValores nulos restantes por columna:")
print(df.isnull().sum())

# ==========================
# 8. Guardar dataset limpio
# ==========================
out_path = "../../data/career_dataset_clean.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

print(f"\nDataset limpio guardado en {out_path}")
print("Shape final:", df.shape)
print("Columnas finales:", df.columns.tolist())
print("\nEjemplo de filas limpias:")
print(df.head(10))
