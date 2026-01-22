import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
from google.auth.exceptions import DefaultCredentialsError
from mlops.data import card_rank, card_suit


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"

    backend = os.environ.get("BACKEND")
    if backend:
        return backend
    try:
        client = run_v2.ServicesClient()
    except DefaultCredentialsError:
        return None

    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_image(image_bytes, filename, content_type, backend):
    predict_url = f"{backend}/classify/"
    files = {"file": (filename, image_bytes, content_type)}
    response = requests.post(predict_url, files=files, timeout=15)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Playing Cards Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        result = classify_image(image_bytes, uploaded_file.name, uploaded_file.type, backend=backend)

        image = uploaded_file.read()

        if result is not None:
            card_suit_name, card_rank_name = result["predicted"]
            probs_suit, probs_rank = result["probabilities"]

            # Show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.header(f"Prediction: {card_rank_name} of {card_suit_name}")

            rank_labels = card_rank[:-1]
            df_rank = pd.DataFrame({"Rank": rank_labels, "Probability": probs_rank})
            df_rank["Rank"] = pd.Categorical(df_rank["Rank"], categories=rank_labels, ordered=True)
            st.bar_chart(df_rank, x="Rank", y="Probability")

            suit_labels = card_suit
            df_suit = pd.DataFrame({"Suit": suit_labels, "Probability": probs_suit})
            df_suit["Suit"] = pd.Categorical(df_suit["Suit"], categories=suit_labels, ordered=True)
            st.bar_chart(df_suit, x="Suit", y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
