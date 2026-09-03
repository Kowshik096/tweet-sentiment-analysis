import os

import mlflow

# Set your tracking URI (env var override; empty string falls back to the
# local sqlite DB, matching the behavior of the production pipeline).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")


def test_mlflow_connection():
    try:
        experiments = mlflow.search_experiments()
        assert experiments is not None, "Failed to retrieve experiments from MLflow"
        print("MLflow connection successful. Experiments:", [e.name for e in experiments])
    except Exception as e:
        raise AssertionError(f"MLflow connection failed: {e}")
