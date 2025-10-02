# balance_reduced.py
import pandas as pd
from imblearn.over_sampling import SMOTE

# ==========================
# 1. Cargar dataset reducido
# ==========================
df = pd.read_csv("../../data/data_train_reduced.csv")
print("Dataset reducido:", df.shape)

# Features y target
features = [c for c in df.columns if c[0] in ["R", "I", "A", "S", "E", "C"]]
X = df[features]
y = df["major_group_reduced"]

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
X_res["major_group_reduced"] = y_res
X_res.to_csv("../../data/data_train_reduced_balanced.csv", index=False)
print("Guardado en ../../data/data_train_reduced_balanced.csv")
