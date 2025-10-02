# load_career_dataset.py
import pandas as pd

# ================================
# 1. Cargar dataset con separador correcto
# ================================
file_path = "../../data/career_dataset.csv"

try:
    df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    print("Dataset cargado correctamente")
    print("Shape:", df.shape)
    print("Columnas:", df.columns.tolist())
    print("\nPrimeras filas:")
    print(df.head())

    # ================================
    # 2. Información general
    # ================================
    print("\nInformación general:")
    print(df.info())

    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    print("\nDistribución de la columna 'Job profession':")
    if "Job profession" in df.columns:
        print(df["Job profession"].value_counts().head(20))

except Exception as e:
    print("Error al cargar el dataset:", e)
