from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

# Initialize FastAPI app
app = FastAPI()

# ---------- Bigram Section ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

# ---------- Embedding Section ----------
nlp = spacy.load("en_core_web_lg")

class EmbeddingRequest(BaseModel):
    word: str

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.word)
    embedding = doc.vector.tolist()
    return {
        "word": request.word,
        "embedding_dim": len(embedding),
        "embedding": embedding[:10]  # 返回前10个数，避免太长
    }

# --- Image Classification Endpoint (CIFAR-10) ---
import os
import torch
from fastapi import UploadFile, File, HTTPException
from PIL import Image
from app.infer import load_model, predict_image

# Path to model weights (make sure the file exists after training)
MODEL_PATH = os.getenv("CIFAR10_MODEL_PATH", "app/models/cnn_cifar10.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at startup (if file exists)
clf_model = None
if os.path.exists(MODEL_PATH):
    try:
        clf_model = load_model(MODEL_PATH, device=DEVICE)
        print(f"[INFO] Loaded CIFAR-10 model from {MODEL_PATH} on {DEVICE}.")
    except Exception as e:
        print(f"[WARN] Could not load model at startup: {e}")

@app.post("/predict_image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Upload an image (jpeg/png). Returns predicted CIFAR-10 class.
    """
    if clf_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please ensure weights file exists on server.")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    result = predict_image(clf_model, image, device=DEVICE)
    return result

# Required import at file top or insert here:
from io import BytesIO
