"""Train the taxi trip-duration model and track the run with MLflow.

Run from the repository root:

    uv run python -m src.train --year 2024 --month 1
"""

import json
import logging
import os
from pathlib import Path

import click
import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from src.preprocess import FEATURES, TARGET, download_data, load_data, preprocess

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "train_metrics.json"

EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the processed data into train and test sets (80/20)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit a DictVectorizer + LinearRegression pipeline."""
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(X_train.to_dict(orient="records"), y_train)
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Return the RMSE of the pipeline on the test set."""
    y_pred = pipeline.predict(X_test.to_dict(orient="records"))
    return float(root_mean_squared_error(y_test, y_pred))


def register_model(model_uri: str, run_id: str) -> str:
    """Register a new model version and point the `production` alias at it."""
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=MODEL_NAME,
            tags={"framework": "scikit-learn"},
            description="Model to predict ride duration in NYC",
        )
    except MlflowException as exc:
        # Reuse the model only if it already exists; re-raise any other MLflow error.
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)
    client.set_registered_model_alias(
        name=MODEL_NAME, version=mv.version, alias="production"
    )
    return mv.version


@click.command()
@click.option("--year", default=2024, show_default=True, help="Year of the data.")
@click.option("--month", default=1, show_default=True, help="Month of the data.")
@click.option("--color", default="green", show_default=True, help="Taxi type.")
@click.option("--developer", default="moumita", show_default=True)
def main(year: int, month: int, color: str, developer: str) -> None:
    """Download data, train the model, log it to MLflow and save it locally."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = download_data(year, month, color)
    df = preprocess(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": developer,
                "dataset": f"{color}-taxi",
                "year": year,
                "month": month,
                "features": FEATURES,
                "target": TARGET,
            }
        )
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        pipeline = train(X_train, y_train)
        rmse = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("rmse", rmse)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        version = register_model(model_info.model_uri, run.info.run_id)

    # Keep a local copy so the API and DVC can use the model without MLflow.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metrics = {
        "rmse": round(rmse, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_version": version,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH} and registered {MODEL_NAME} v{version}")


if __name__ == "__main__":
    main()
