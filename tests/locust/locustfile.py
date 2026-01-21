import random
import io
from turtle import color
from PIL import Image
from locust import HttpUser, between, task
from numpy import size


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def classify_image(self) -> None:
        """A task that simulates a user sending a random image to the FastAPI app."""
        
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        img = Image.new("RGB", size=(224, 224), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        filename, mimetype = "test.png", "image/png"  # Replace with actual image data
        files = {"file": (filename, buf, mimetype)}
        # Use catch_response so we can mark failures when JSON/fields aren't as expected
        with self.client.post("/classify/", files=files, timeout=15, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            # Basic schema check (adjust to your real response)
            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"Invalid JSON: {e}")
                return

            # Example validations—tweak to match your API contract
            if "prediction" not in data:
                resp.failure(f"Missing 'prediction' in response: {data}")
            else:
                resp.success()
