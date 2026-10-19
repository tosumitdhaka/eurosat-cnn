"""FastAPI service: POST /predict with an image file -> predicted EuroSAT land-use class."""
import io, json
from pathlib import Path
import numpy as np, onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
META = json.load(open(MODEL_DIR / "meta.json"))
CLASSES, SIZE = META["classes"], META["size"]
SESSION = ort.InferenceSession(str(MODEL_DIR / "eurosat_resnet18.onnx"), providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

app = FastAPI(title="EuroSAT land-use classifier")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((64, 64), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return (x - MEAN) / STD


def predict_image(img: Image.Image, top_k: int = 3) -> dict:
    logits = SESSION.run(None, {"image": preprocess(img)})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    order = np.argsort(-p)
    return {"prediction": CLASSES[order[0]], "confidence": float(p[order[0]]),
            "top": [{"class": CLASSES[i], "probability": float(p[i])} for i in order[:top_k]]}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")
    return predict_image(img)
