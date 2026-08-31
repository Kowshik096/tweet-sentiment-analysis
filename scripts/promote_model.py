import os
import mlflow

def promote_model():
    # Set up MLflow tracking URI (env var override, local sqlite default)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))

    client = mlflow.MlflowClient()

    model_name = "tweet_sentiment_model"
    # Get the latest version in staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise RuntimeError(f"No model found in 'Staging' stage for '{model_name}'. Cannot promote.")
    latest_version_staging = staging_versions[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()
