# balance_dataset.py
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ==========================
# 1. Cargar dataset limpio
# ==========================
df = pd.read_csv("../../data/data_train_clean.csv")
print("Dataset original:", df.shape)

# Features y target
features = [c for c in df.columns if c[0] in ["R", "I", "A", "S", "E", "C"]]
X = df[features]
y = df["major_group"]

print("Distribución original:")
print(y.value_counts(normalize=True) * 100)

# ==========================
# 2. Aplicar SMOTE
# ==========================
smote = SMOTE(random_state=42, sampling_strategy="auto")
X_res, y_res = smote.fit_resample(X, y)

print("Dataset balanceado:", X_res.shape)
print("Distribución balanceada:")
print(y_res.value_counts(normalize=True) * 100)

# ==========================
# 3. Guardar dataset balanceado
# ==========================
X_res["major_group"] = y_res
X_res.to_csv("../../data/data_train_balanced.csv", index=False)
print("Guardado en ../../data/data_train_balanced.csv")
