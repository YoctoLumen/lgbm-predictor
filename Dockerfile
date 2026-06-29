# Image de base légère
FROM python:3.11-slim

# Installation des dépendances système nécessaires à LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY service.py .
COPY model_lgbm/ ./model_lgbm/

# Exposition du port
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
