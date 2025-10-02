import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================
# 1. Cargar datasets
# ==========================
train_df = pd.read_csv("../../data/data_train_reduced_balanced.csv")  # balanceado y reducido
test_df = pd.read_csv("../../data/data_test_reduced.csv")             # reducido (sin balancear)

print("Train:", train_df.shape, " Test:", test_df.shape)

# ==========================
# 2. Calcular los puntajes RIASEC
# ==========================
def calcular_scores(df):
    categorias = ["R", "I", "A", "S", "E", "C"]
    scores = {cat: df[[c for c in df.columns if c.startswith(cat)]].sum(axis=1) for cat in categorias}
    return pd.DataFrame(scores)

def calcular_riasec(df):
    scores_df = calcular_scores(df)
    return scores_df.idxmax(axis=1)  # retorna la letra dominante

# ==========================
# 3. Dataset RIASEC (solo 6 features)
# ==========================
X_train = calcular_scores(train_df)   # 6 columnas R, I, A, S, E, C
y_train_riasec = calcular_riasec(train_df)

X_test = calcular_scores(test_df)
y_test_riasec = calcular_riasec(test_df)

print("Features usadas:", X_train.columns.tolist())
print("Ejemplo de etiquetas RIASEC:", y_train_riasec.unique())

# ==========================
# 4. MODELO: Perfil RIASEC
# ==========================
model_riasec = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model_riasec.fit(X_train, y_train_riasec)

# ==========================
# 5. Evaluación
# ==========================
y_pred_riasec = model_riasec.predict(X_test)

print("\nReporte de clasificación (RIASEC):")
print(classification_report(y_test_riasec, y_pred_riasec))
print("Accuracy RIASEC:", accuracy_score(y_test_riasec, y_pred_riasec))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_riasec, y_pred_riasec),
            annot=True, cmap="Blues", fmt="d",
            xticklabels=model_riasec.classes_,
            yticklabels=model_riasec.classes_)
plt.title("Matriz de Confusión RIASEC")
plt.show()

# ==========================
# 6. Guardar modelo RIASEC
# ==========================
joblib.dump(model_riasec, "../../models/riasec_model.pkl")
print("Modelo RIASEC guardado en ../../models/riasec_model.pkl")
