import io, sys
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app, CLASSES  # noqa: E402

client = TestClient(app)


def test_predict_returns_valid_class_and_probabilities():
    arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG")
    r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] in CLASSES and 0 < j["confidence"] <= 1
    assert sum(t["probability"] for t in j["top"]) <= 1.0001


def test_bad_upload_is_400():
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
