# preprocess_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Cargar dataset limpio
# ---------------------------
df = pd.read_csv("../../data/data_clean.csv", low_memory=False)
print("Dataset original:", df.shape)

# ---------------------------
# 2. Filtrar categorías
# ---------------------------
# Quitar unspecified
df = df[df["major_group"] != "unspecified"]

# Reagrupar categorías pequeñas en "other"
min_threshold = 0.01  # 1% del total
counts = df["major_group"].value_counts(normalize=True)
rare_categories = counts[counts < min_threshold].index
df["major_group"] = df["major_group"].replace(rare_categories, "other")

print("Distribución final de categorías (%):")
print(df["major_group"].value_counts(normalize=True) * 100)

# ---------------------------
# 3. Dividir train/test
# ---------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["major_group"],
    random_state=42
)

print("Train:", train_df.shape, " Test:", test_df.shape)

# ---------------------------
# 4. Guardar
# ---------------------------
train_df.to_csv("../../data/data_train_clean.csv", index=False)
test_df.to_csv("../../data/data_test_clean.csv", index=False)

print("Guardados: data_train_clean.csv y data_test_clean.csv")
