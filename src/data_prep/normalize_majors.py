# limpiar_carreras.py
import pandas as pd
import unicodedata
from rapidfuzz import process

# ---------------------------
# Función: limpieza básica
# ---------------------------
def limpiar_texto(x):
    if pd.isna(x): 
        return None
    x = str(x).strip().lower()
    x = ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn')  # elimina acentos
    return x

# ---------------------------
# Diccionario de categorías amplias
# ---------------------------
categorias = {
    "health": ["medicine", "nursing", "dentistry", "health", "biomedical", "pharmacy", "nutrition"],
    "engineering": ["engineering", "mechanical", "civil", "electrical", "mechatronic", "industrial"],
    "business": ["business", "management", "account", "marketing", "commerce", "finance", "mba", "administration"],
    "social_sciences": ["psychology", "sociology", "anthropology", "criminology", "political", "international relations"],
    "law": ["law", "legal", "criminology", "justice"],
    "education": ["education", "teaching", "pedagogy"],
    "arts": ["art", "design", "music", "theatre", "film", "fashion"],
    "sciences": ["biology", "chemistry", "physics", "mathematics", "math", "science", "geology"],
    "it_computing": ["computer", "informatics", "cyber", "it", "software", "programming", "technology"],
    "communication": ["communication", "journalism", "media"],
    "economics": ["economics", "economy", "economic"],
    "agriculture": ["agriculture", "veterinary", "zoology", "animal", "botany"],
    "architecture": ["architecture", "architect", "urbanism"],
    "humanities": ["history", "literature", "philosophy", "linguistics", "languages", "humanities"],
    "sports": ["sport", "physical education", "kinesiology"],
    "unspecified": ["idk", "not sure", "none", "-", "no", "undecided", "undeclared", "nil", "na"]
}

# ---------------------------
# Función: asignar categoría
# ---------------------------
def asignar_categoria(carrera):
    if not carrera:
        return "unspecified"
    for cat, keywords in categorias.items():
        for kw in keywords:
            if kw in carrera:
                return cat
    return "other"

# ---------------------------
# Script principal
# ---------------------------
if __name__ == "__main__":
    # 1. Cargar dataset bruto
    df = pd.read_csv("../../data/data.csv", sep=None, engine="python")
    print("Dataset bruto:", df.shape)

    # 2. Limpiar la columna major
    df["major_clean"] = df["major"].apply(limpiar_texto)

    # 3. Asignar categorías
    df["major_group"] = df["major_clean"].apply(asignar_categoria)

    # 4. Resumen
    print("\nEjemplo de mapeo:")
    print(df[["major", "major_group"]].head(20))

    print("\nCantidad de categorías:")
    print(df["major_group"].value_counts())

    # 5. Guardar dataset limpio
    df.to_csv("../../data/data_clean.csv", index=False)
    print("\nGuardado en ../../data/data_clean.csv con columna 'major_group'")
