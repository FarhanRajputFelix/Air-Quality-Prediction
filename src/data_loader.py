import zipfile
from pathlib import Path
import requests
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
ZIP_NAME = "AirQualityUCI.zip"
CSV_NAME = "AirQualityUCI.csv"


def download_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"Downloading dataset to {zip_path}")
        response = requests.get(DATA_URL, stream=True, timeout=30)
        response.raise_for_status()
        with open(zip_path, "wb") as stream:
            for chunk in response.iter_content(chunk_size=8192):
                stream.write(chunk)
    return zip_path


def extract_dataset(zip_path: Path, data_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(path=data_dir)
    return data_dir / CSV_NAME


def load_raw_dataset(data_dir: Path) -> pd.DataFrame:
    zip_path = data_dir / ZIP_NAME
    if not zip_path.exists():
        zip_path = download_dataset(data_dir)
    csv_path = data_dir / CSV_NAME
    if not csv_path.exists():
        csv_path = extract_dataset(zip_path, data_dir)

    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        na_values=[-200, "-200"],
    )
    df = df.dropna(how="all", axis=1)
    df.columns = [col.strip() for col in df.columns]

    if "Date" in df.columns and "Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce",
        )
    return df


def get_dataset(data_dir: Path = Path("data")) -> pd.DataFrame:
    data_dir = Path(data_dir)
    df = load_raw_dataset(data_dir)
    return df
