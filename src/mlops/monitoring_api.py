import os
from pathlib import Path
import datetime
import pandas as pd

# from evidently.legacy.metric_preset import TargetDriftPreset
# from evidently.legacy.report import Report
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from google.oauth2 import service_account
from mlops.data import load_data
from mlops.datadrift import extract_image_features
import torch
import torchvision
from contextlib import asynccontextmanager


BUCKET_NAME = "dtu-mlops-group-48-data"


def get_gcs_client():
    """Get GCS client using service account file or default credentials."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(credentials=credentials, project="dtu-mlops-group-48")
    return storage.Client(project="dtu-mlops-group-48")


def save_to_gcp(file: str):
    """Save the prediction results and input image to GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)
    # Upload the html file
    image_blob = bucket.blob(f"reports/api_monitoring_{time}.html")
    image_blob.upload_from_string(file)

    print("Prediction and input image saved to GCP bucket.")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> str:
    """Run the analysis and return the report."""
    try:
        text_overview_report = Report(
            metrics=[DataDriftPreset(), DataSummaryPreset()],
            include_tests=True,
        )
        result = text_overview_report.run(
            reference_data=reference_data,
            current_data=current_data,
        )
        result.save_html("report.html")
        with open("report.html", "r") as file:
            html_str = file.read()
    except ValueError as e:
        if "Too many bins" in str(e):
            text_overview_report = Report(
                metrics=[DataSummaryPreset()],
                include_tests=False,
            )
            result = text_overview_report.run(
                reference_data=reference_data,
                current_data=current_data,
            )
            result.save_html("report.html")
            with open("report.html", "r") as file:
                html_str = file.read()
        else:
            raise

    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("reports/api_monitoring_report.html")
    blob.upload_from_string(html_str, content_type="text/html")
    return html_str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data
    os.makedirs(os.path.dirname("data/"), exist_ok=True)
    download_from_gcp("data/processed/train.pt", "data/processed/train.pt")
    training_data_dataset = load_data(processed_dir="data/processed", split="train")
    training_data = extract_image_features(training_data_dataset)

    yield

    del training_data


app = FastAPI(lifespan=lifespan)


def download_from_gcp(gcs_path, local_path):
    """Download the model from GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


def load_latest_files(directory: Path, n: int) -> torch.utils.data.Dataset:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n, directory=str(directory / "predictions"))

    # Get all prediction files in the directory
    files = (directory / "predictions").glob("input_*.jpg")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    resize = torchvision.transforms.Resize((224, 224))

    images = []
    for file in latest_files:
        img_bytes = torchvision.io.read_file(str(file))
        img = torchvision.io.decode_image(img_bytes)
        if img.size(0) == 1:
            img = img.expand(3, -1, -1)
        elif img.shape[0] == 4:
            # RGBA → drop alpha
            img = img[:3, :, :]

        elif img.shape[0] != 3:
            # Unknown case: convert using PIL fallback
            from torchvision.transforms.functional import to_pil_image, to_tensor

            img = to_tensor(to_pil_image(img))  # guaranteed 3 channel

        images.append(resize(img))

    labels = [0] * len(images)

    dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))

    return dataset


def download_files(n: int = 5, directory: str = "predictions") -> None:
    """Download the N latest prediction files from the GCP bucket.

    Args:
        n: Number of latest files to download.
        directory: Directory path where files will be saved.
    """
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="predictions/input_"))
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
    prediction_data = extract_image_features(prediction_data)
    html_str = run_analysis(training_data, prediction_data)

    return HTMLResponse(content=html_str, status_code=200)
