FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y crear carpeta de modelos
COPY src/ /app/src
RUN mkdir -p /app/models

# Variable de entorno
ENV MODELS_DIR=/app/models

# Exponer puerto
EXPOSE 8000

# Ejecutar con un solo worker para evitar OOM en Render Free
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "src.inference.api:app", "-b", "0.0.0.0:8000"]
