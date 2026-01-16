from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from mlops.model import Model
from mlops.data import card_suit, card_rank
from torchvision import transforms

card_classes = {"suit": card_suit, "rank": card_rank}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, transform, card_classes
    # Load model
    model = Model()
    model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model.mean, std=model.std),  # maybe no normalization
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield

    # Cleaning up
    del model, device, transform, card_classes


app = FastAPI(lifespan=lifespan, debug=True)


def predict_card(image_path: str) -> str:
    """Predict card class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
    _, predicted_suit_idx = torch.max(output["suit"], 1)
    _, predicted_rank_idx = torch.max(output["rank"], 1)
    return (torch.softmax(output["suit"], 1), torch.softmax(output["rank"], 1)), (
        card_classes["suit"][predicted_suit_idx.item()],
        card_classes["rank"][predicted_rank_idx.item()],
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}  # Idk man endpoint, maybe diff message??


# FastAPI endpoint for card classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        # async with await anyio.open_file(file.filename, "wb") as f:
        with open(file.filename, "wb") as f:
            f.write(contents)
        probabilities, prediction = predict_card(file.filename)
        return {"filename": file.filename, "predicted ": prediction, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # from e
