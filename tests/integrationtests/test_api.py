from fastapi.testclient import TestClient
from mlops.backend import app, predict_card
from io import BytesIO
from PIL import Image

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from the backend!"}


# testing if the model imports images correctly
def test_image_load():
    with TestClient(app) as clients:
        ## Create test image in memory
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("test.png", buf.read(), "image/png")}
        ## Check if image is loaded correctly and prediction endpoint works
        response = clients.post("/classify/", files=files)
        print(response.json())
        assert response.status_code == 200
        assert tuple(response.json()["predicted"]) == predict_card(img)[1]
