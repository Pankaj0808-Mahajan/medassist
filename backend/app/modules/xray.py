"""Chest X-ray classifier: fine-tuned DenseNet121 (NORMAL vs PNEUMONIA) with Grad-CAM heatmap."""
import base64
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..config import WEIGHTS

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def preprocess(gray: np.ndarray):
    """CLAHE -> 224x224 -> 3 channels -> ImageNet normalisation. Identical to colab/02_xray_densenet.py."""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    g = cv2.resize(g, (224, 224)).astype(np.float32) / 255.0
    x = (np.stack([g, g, g]) - MEAN) / STD
    return torch.from_numpy(x).unsqueeze(0).float(), g


class XrayClassifier:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        path = WEIGHTS / "xray_densenet121.pt"
        if path.exists():
            m = models.densenet121(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            self.model = m.eval()
            meta = WEIGHTS / "xray_meta.json"
            if meta.exists():
                self.threshold = float(json.loads(meta.read_text()).get("threshold", 0.5))

    def predict(self, data: bytes) -> dict:
        if self.model is None:
            return {"available": False, "note": "No xray_densenet121.pt weights found in backend/weights."}
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read the X-ray image (use JPEG or PNG).")
        x, base = preprocess(gray)
        x.requires_grad_(True)
        with torch.enable_grad():  # Grad-CAM needs gradients w.r.t. the last feature map
            feats = self.model.features(x)
            feats.retain_grad()
            logits = self.model.classifier(F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1))
            p = float(F.softmax(logits, 1)[0, 1])
            cls = 1 if p >= self.threshold else 0
            logits[0, cls].backward()
        w = feats.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1))[0].detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
        png = cv2.imencode(".png", cv2.resize(overlay, (448, 448)))[1].tobytes()
        return {"available": True, "finding": "PNEUMONIA" if cls == 1 else "NORMAL", "p_pneumonia": p,
                "confidence": max(p, 1 - p), "threshold": self.threshold,
                "heatmap": "data:image/png;base64," + base64.b64encode(png).decode(),
                "model": "DenseNet121 fine-tuned (Colab)"}
