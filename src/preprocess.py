"""Download, load and clean the NYC taxi trip data used for training."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def get_data_path(year: int, month: int, color: str = "green") -> Path:
    """Return the local path of one monthly trip-data file."""
    return DATA_DIR / f"{color}_tripdata_{year}-{month:02d}.parquet"


def download_data(year: int, month: int, color: str = "green") -> Path:
    """Download one month of trip data unless it is already on disk."""
    data_path = get_data_path(year, month, color)
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            f"{DATA_URL}/{color}_tripdata_{year}-{month:02d}.parquet", data_path
        )
    return data_path


def load_data(data_path: Path) -> pd.DataFrame:
    """Read a trip-data parquet file."""
    return pd.read_parquet(data_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate duration, remove outliers, and treat location IDs as categories."""
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)].copy()
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    return df[FEATURES + [TARGET]]
