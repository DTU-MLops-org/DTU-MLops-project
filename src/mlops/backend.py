from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO


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
            # transforms.Normalize(mean=model.mean, std=model.std),  # normalization has to be the same as during training
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield

    # Cleaning up
    del model, device, transform, card_classes


app = FastAPI(lifespan=lifespan, debug=True)


def predict_card(image: Image.Image) -> str:
    """Predict card class (or classes) given image path and return the result."""
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    suit_probs = torch.softmax(output["suit"], 1).cpu().tolist()
    rank_probs = torch.softmax(output["rank"], 1).cpu().tolist()

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
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        probabilities, prediction = predict_card(image)
        return {"filename": file.filename, "predicted ": prediction, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # from e
