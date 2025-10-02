# reduce_categories.py
import pandas as pd

# ==========================
# 1. Cargar datasets
# ==========================
train_df = pd.read_csv("../../data/data_train_clean.csv")
test_df = pd.read_csv("../../data/data_test_clean.csv")

print("Antes:", train_df["major_group"].unique())

# ==========================
# 2. Definir mapping
# ==========================
mapping = {
    "arts": "arts_communication",
    "communication": "arts_communication",
    "business": "business",
    "economics": "business",  # fusionado
    "sciences": "science_tech",
    "it_computing": "science_tech",
    "engineering": "science_tech",
    "health": "health",
    "education": "education",
    "law": "law",
    "social_sciences": "social_humanities",
    "humanities": "social_humanities",
    "other": "other"
}

# ==========================
# 3. Aplicar mapping
# ==========================
train_df["major_group_reduced"] = train_df["major_group"].map(mapping)
test_df["major_group_reduced"] = test_df["major_group"].map(mapping)

print("Después:", train_df["major_group_reduced"].unique())

# ==========================
# 4. Guardar datasets
# ==========================
train_df.to_csv("../../data/data_train_reduced.csv", index=False)
test_df.to_csv("../../data/data_test_reduced.csv", index=False)

print("Guardados como data_train_reduced.csv y data_test_reduced.csv")
