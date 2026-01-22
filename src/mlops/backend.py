from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from PIL import Image
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
import datetime
import os
import json

from mlops.model import Model
from mlops.data import card_suit, card_rank
from torchvision import transforms

card_classes = {"suit": card_suit, "rank": card_rank}

BUCKET_NAME = "dtu-mlops-group-48-data"
MODEL_FILE_NAME = "models/model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, transform, card_classes
    # Load model
    model = download_model_from_gcp()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=model.mean, std=model.std),  # normalization has to be the same as during training
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield

    # Cleaning up
    del model, device, transform, card_classes


app = FastAPI(lifespan=lifespan, debug=True)


def get_gcs_client():
    """Get GCS client using service account file or default credentials."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(credentials=credentials, project="dtu-mlops-group-48")
    return storage.Client(project="dtu-mlops-group-48")


def download_model_from_gcp():
    """Download the model from GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    checkpoint_bytes = blob.download_as_bytes()
    checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu")
    model = Model()
    model.load_state_dict(checkpoint)
    print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME}.")
    return model


# Save prediction results to GCP
def save_prediction_to_gcp(filename: str, image_bytes: bytes, probabilities: list[float], prediction: str):
    """Save the prediction results and input image to GCP bucket."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)
    # Upload the input image
    image_blob = bucket.blob(f"predictions/input_{time}.jpg")
    image_blob.upload_from_string(image_bytes, content_type="image/jpeg")

    # Prepare prediction data
    data = {
        "file": filename,
        "image_path": f"predictions/input_{time}.jpg",
        "probabilities": probabilities,
        "prediction": prediction,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    data_blob = bucket.blob(f"predictions/prediction_{time}.json")
    data_blob.upload_from_string(json.dumps(data))
    print("Prediction and input image saved to GCP bucket.")


def predict_card(image: Image.Image) -> str:
    """Predict card class (or classes) given image path and return the result."""
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    suit_probs = torch.softmax(output["suit"], 1).cpu().squeeze(0).tolist()
    rank_probs = torch.softmax(output["rank"], 1).cpu().squeeze(0).tolist()

    suit_logits = output["suit"]  # shape [1, num_suits]
    rank_logits = output["rank"]  # shape [1, num_ranks]

    predicted_suit_idx = int(torch.argmax(suit_logits, dim=1).item())
    predicted_rank_idx = int(torch.argmax(rank_logits, dim=1).item())

    card_suit_str = card_classes["suit"][predicted_suit_idx]
    card_rank_str = card_classes["rank"][predicted_rank_idx]

    return (suit_probs, rank_probs), (card_suit_str, card_rank_str)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}  # Idk man endpoint, maybe diff message??


# FastAPI endpoint for card classification
@app.post("/classify")
async def classify_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        probabilities, prediction = predict_card(image)

        background_tasks.add_task(save_prediction_to_gcp, file.filename, contents, probabilities, prediction)

        return {"filename": file.filename, "predicted": prediction, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # from e
