import os

import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set your tracking URI (env var override; empty string falls back to the
# local sqlite DB, matching the behavior of the production pipeline).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")


@pytest.mark.parametrize(
    "model_name, stage",
    [
        ("tweet_sentiment_model", "staging"),
    ],
)
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(
            f"Model '{model_name}' version {latest_version} loaded "
            f"successfully from '{stage}' stage."
        )

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
