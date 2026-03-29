import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from timm.models.swin_transformer import SwinTransformer
import numpy as np
from pathlib import Path

# ==========================================================
# 📂 DYNAMIC PATHING (Fixes Linux/Windows Mismatch)
# ==========================================================
# This finds the 'project_root' folder absolute path
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_model", "swin_isic_phase2_full_model.pth")

# 1. Enable safe loading for timm models
torch.serialization.add_safe_globals([SwinTransformer])

# 2. Global Model Variable
model = None

# Load model once globally on startup
if not os.path.exists(MODEL_PATH):
    print(f"❌ CRITICAL ERROR: Model file not found at: {MODEL_PATH}")
    print(f"Current Directory: {os.getcwd()}")
else:
    try:
        # Load to CPU (Standard for Streamlit Free Tier)
        # Using weights_only=False because custom timm models often require full pickling
        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
        print(f"✅ SUCCESS: Swin-Transformer loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ FATAL: Model loading failed: {e}")
        model = None

# ==========================================================
# 🖼️ IMAGE PRE-PROCESSING
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# STRICT CLASS MAPPING (Ensure this matches your training order)
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


def predict_image(image_path: str):
    """
    Performs inference on a single image.
    Returns: Dict containing top-3 predictions or error.
    """
    if model is None:
        return {
            "error": "Model not loaded properly on server. Check logs for pathing errors.",
            "path_attempted": str(MODEL_PATH)
        }

    try:
        # Load and convert to RGB (Fixes PNG alpha channel issues)
        img_pil = Image.open(image_path).convert("RGB")

        # Apply transforms and add batch dimension
        input_tensor = transform(img_pil).unsqueeze(0).float()

        with torch.no_grad():
            outputs = model(input_tensor)

            # Handle list/tuple outputs (common in specialized timm heads)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            # Global Average Pooling if the model returns 4D feature maps
            if logits.ndim > 2:
                logits = torch.mean(logits, dim=(2, 3))

            logits = logits.squeeze()

            # Calculate Probabilities
            probs = F.softmax(logits, dim=0)

        # Extract Top 3 results
        top3_prob, top3_idx = torch.topk(probs, 3)

        results = []
        for i in range(3):
            idx = top3_idx[i].item()
            results.append({
                "label": CLASS_NAMES[idx],
                "confidence": float(top3_prob[i].item())
            })

        return {
            "label": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top_3": results
        }
    except Exception as e:
        print(f"⚠️ Inference Error: {str(e)}")
        return {"error": f"Inference Error: {str(e)}"}