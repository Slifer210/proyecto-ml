FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelos
COPY src/ /app/src
RUN mkdir -p /app/models

# Definir variable de entorno para la ruta de modelos
ENV MODELS_DIR=/app/models

# Exponer puerto
EXPOSE 8000

# Correr con Gunicorn + Uvicorn en producción
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.inference.api:app", "-b", "0.0.0.0:8000", "--workers", "2"]
