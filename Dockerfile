FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances système utiles (compilation éventuelle de wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Étape séparée pour profiter du cache Docker
COPY requirements.txt .

# 1) Upgrade pip/setuptools/wheel (plus robuste TLS + wheels précompilées)
RUN python -m pip install --upgrade pip setuptools wheel

# 2) Installer les deps (timeout plus grand)
RUN python -m pip install --default-timeout=120 -r requirements.txt

# Copie du code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
