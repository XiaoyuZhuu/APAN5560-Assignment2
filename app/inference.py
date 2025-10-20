# app/inference.py
from io import BytesIO
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from .cnn import SimpleCNN

_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

class Predictor:
    def __init__(self, ckpt_path: str = "models/cnn_cifar10.pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.model = SimpleCNN(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes):
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = _transform(img).unsqueeze(0)  # 1x3x64x64
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        return {
            "class_idx": int(idx.item()),
            "class_name": self.classes[int(idx.item())],
            "confidence": float(conf.item())
        }
