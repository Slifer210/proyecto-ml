import pandas as pd

# Cargar dataset limpio
df = pd.read_csv("../../data/data_clean.csv")

# Conteo de categorías
conteo = df["major_group"].value_counts(normalize=True) * 100
print("Distribución de categorías (%):")
print(conteo)

# Ver ejemplos de 'unspecified'
print("\nEjemplos de registros 'unspecified':")
print(df[df["major_group"] == "unspecified"]["major"].head(20))
