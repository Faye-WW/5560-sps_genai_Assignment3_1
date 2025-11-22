# app/inference.py
from __future__ import annotations
from pathlib import Path
import io
from typing import Dict

import torch
import torchvision.transforms as T
from PIL import Image

from helper_lib.model import get_model

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = get_model("CNN", num_classes=len(CIFAR10_CLASSES)).to(_device)

# weight
_ckpt = Path("models/cnn.pt")
if not _ckpt.exists():
    raise RuntimeError("models/cnn.pt not found. Please train first: python -m app.train_cnn")
_state = torch.load(_ckpt, map_location=_device)
_model.load_state_dict(_state)
_model.eval()

_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

def predict_image_bytes(image_bytes: bytes) -> Dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(prob).item())
    return {
        "class": CIFAR10_CLASSES[idx],
        "index": idx,
        "probs": [float(p) for p in prob.cpu().tolist()]
    }
