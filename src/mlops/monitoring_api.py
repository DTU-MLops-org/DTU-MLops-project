import os
from pathlib import Path
import datetime
import anyio
import pandas as pd
from evidently.legacy.metric_preset import TargetDriftPreset, TextEvals
from evidently.legacy.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from google.oauth2 import service_account
from mlops.data import load_data
from mlops.datadrift import extract_image_features
import torch
import torchvision


BUCKET_NAME = "gcp_monitoring_exercise"


def get_gcs_client():
    """Get GCS client using service account file or default credentials."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(credentials=credentials, project="dtu-mlops-group-48")
    return storage.Client(project="dtu-mlops-group-48")


def save_to_gcp(file: str, probabilities: list[float], prediction: str):
    """Save the prediction results and input image to GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)
    # Upload the html file
    image_blob = bucket.blob(f"reports/api_monitoring_{time}.html")
    image_blob.upload_from_string(file)

    print("Prediction and input image saved to GCP bucket.")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="content"), TargetDriftPreset(columns=["sentiment"])])
    result = text_overview_report.run(reference_data=reference_data, current_data=current_data)
    html_str = result.get_html()
    save_to_gcp(file=html_str)


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data
    os.makedirs(os.path.dirname("data/"), exist_ok=True)
    download_from_gcp("data/processed/train.pt", "data/train.pt")
    training_data_dataset = load_data(processed_dir="data/processed/train.pt", split="train")
    training_data = extract_image_features(training_data_dataset)

    yield

    del training_data


def download_from_gcp(gcs_path, local_path):
    """Download the model from GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n, directory=str(directory / "predictions"))

    # Get all prediction files in the directory
    files = directory.glob("predictions/input_*.jpg")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed

    images = []
    for file in latest_files:
        img_bytes = torchvision.io.read_file(str(file))
        data = torchvision.io.decode_image(img_bytes)
        images.append(data)

    labels = [0] * len(images)

    dataset = torch.utils.data.TensorDataset(images, labels)

    return dataset


def download_files(n: int = 5, directory: str = "predictions") -> None:
    """Download the N latest prediction files from the GCP bucket.

    Args:
        n: Number of latest files to download.
        directory: Directory path where files will be saved.
    """
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="predictions/input_")
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    os.makedirs(directory, exist_ok=True)
    for blob in latest_blobs:
        local_path = os.path.join(directory, os.path.basename(blob.name))
        blob.download_to_filename(local_path)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
