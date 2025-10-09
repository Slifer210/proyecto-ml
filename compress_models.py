import os
import joblib

def compress_model(file_path, compress_level=3):
    """Carga un modelo .pkl y lo guarda comprimido."""
    try:
        print(f"\nProcesando: {file_path}")
        model = joblib.load(file_path)

        # Reemplazar directamente el archivo original
        temp_path = file_path + ".tmp"
        joblib.dump(model, temp_path, compress=compress_level)

        old_size = os.path.getsize(file_path) / (1024 * 1024)
        new_size = os.path.getsize(temp_path) / (1024 * 1024)
        reduction = 100 * (1 - new_size / old_size)

        os.replace(temp_path, file_path)
        print(f"Comprimido: {os.path.basename(file_path)}")
        print(f"Tamaño antes: {old_size:.1f} MB → después: {new_size:.1f} MB ({reduction:.1f}% menos)")

    except Exception as e:
        print(f"Error al procesar {file_path}: {e}")


def main():
    project_root = os.getcwd()
    models_dir = os.path.join(project_root, "models")

    if not os.path.exists(models_dir):
        print("No se encontró la carpeta 'models' en el proyecto.")
        return

    pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print("No hay archivos .pkl para comprimir.")
        return

    print(f"Se encontraron {len(pkl_files)} modelos en /models:")
    for f in pkl_files:
        print(f" - {f}")

    for f in pkl_files:
        compress_model(os.path.join(models_dir, f))


if __name__ == "__main__":
    main()
