# ML Inference API (FastAPI + Docker) with MLflow


API d’inférence **FastAPI** conteneurisée (**Docker**) + **MLflow** pour le tracking. 
Le modèle (RandomForest sur Iris) est entraînable et versionné localement.


## Lancer rapidement
```bash
# (optionnel) Entraîner + logger dans MLflow
python -m pip install -r requirements.txt
python train/train.py


# Builder l'image et lancer l'API
docker build -t ml-inference-api:latest .
docker run -p 8000:8000 ml-inference-api:latest