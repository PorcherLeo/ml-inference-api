# train/train.py
from __future__ import annotations
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect
import numpy as np
from mlflow.models import infer_signature

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# MLflow en file store local (UI via docker-compose)
MLFLOW_DIR = os.path.join(ROOT, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("iris-random-forest")

def train_and_log(random_state: int = 42, n_estimators: int = 200, max_depth: int | None = None):
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    with mlflow.start_run() as run:
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        # Signature + input_example (pour éviter les warnings MLflow)
        signature = infer_signature(X_train, clf.predict(X_train))
        input_example = X_train[:1].astype(np.float32)

        # Un SEUL log_model — compatible toutes versions
        log_kwargs = dict(sk_model=clf, signature=signature, input_example=input_example)
        if "name" in inspect.signature(mlflow.sklearn.log_model).parameters:
            mlflow.sklearn.log_model(name="model", **log_kwargs)          # MLflow récent
        else:
            mlflow.sklearn.log_model(artifact_path="model", **log_kwargs) # fallback

        # Export pour l’API FastAPI
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        joblib.dump(clf, model_path)

        print(f"Model saved to {model_path}")
        print(f"Run ID: {run.info.run_id}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_and_log()
