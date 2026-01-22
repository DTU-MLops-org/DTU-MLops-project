from mlops.data import load_data, preprocess_data

import torch
import argparse
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger


## Initial setup, load models
# -----------------------------
# Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None


def _load_clip_model():
    """Lazy load CLIP model and processor."""
    global model, processor
    if model is None or processor is None:
        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/tmp/huggingface_cache",
        ).to(device)
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/tmp/huggingface_cache",
        )
    return model, processor


# -----------------------------
# Extract image features
# -----------------------------
def extract_image_features(dataset, max_samples=2000, batch_size=64, extracted_features=50):
    """Extract image features using CLIP model."""
    model, processor = _load_clip_model()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []

    with torch.no_grad():
        seen = 0
        for images, _ in loader:
            if seen >= max_samples:
                break

            inputs = processor(images=images, return_tensors="pt", do_rescale=True).to(device)
            f = model.get_image_features(**inputs)
            f = torch.nn.functional.normalize(f, dim=1)

            feats.append(f.cpu())
            seen += images.size(0)

    features = torch.cat(feats).numpy()
    df_features = pd.DataFrame(features[:, :extracted_features])
    df_features.columns = [f"feature_{i}" for i in range(df_features.shape[1])]
    return df_features


def run_data_drift(angle: float):
    # Rotate the images if needed
    logger.info(f"Fetching and rotating images {angle} degrees...")
    preprocess_data(rotate=True, angle=angle)

    regular_data = load_data(split="train")
    rotated_data = load_data(processed_dir="data/processed_rotated", split="train")

    logger.info("Extracting image features...")
    df_regular = extract_image_features(regular_data)
    df_rotated = extract_image_features(rotated_data)

    logger.info("Generating data drift report...")
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()], include_tests=True)
    report_html = report.run(reference_data=df_regular, current_data=df_rotated)

    report_html.save_html(f"reports/datadrift/rotation_{angle}_degrees.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, required=True)
    args = parser.parse_args()

    run_data_drift(angle=args.angle)
