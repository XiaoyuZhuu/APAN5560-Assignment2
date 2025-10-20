# app/infer.py
# Image inference utilities for the trained CNN.

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from app.cnn_model import CNN64

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_LABELS = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def load_model(weights_path: str, device: str = "cpu") -> CNN64:
    model = CNN64(num_classes=10)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model

def predict_image(model: CNN64, image: Image.Image, device: str = "cpu"):
    # Convert PIL image to model tensor
    x = _transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        return {
            "class_index": pred_idx,
            "class_name": CIFAR10_LABELS[pred_idx],
            "probability": float(probs[pred_idx].item())
        }
