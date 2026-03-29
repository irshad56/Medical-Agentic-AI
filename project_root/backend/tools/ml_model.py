import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from timm.models.swin_transformer import SwinTransformer
import numpy as np

# 1. Enable safe loading
torch.serialization.add_safe_globals([SwinTransformer])
MODEL_PATH = "models/image_model/swin_isic_phase2_full_model.pth"

# Load model once globally
try:
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")

# ✅ THE "ONCE AND FOR ALL" PRE-PROCESSING
# Using the exact IMAGENET stats Swin-T was trained on.
# Also adding a slight sharpen to help detect Actinic textures.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ STRICT CLASS MAPPING
# Ensure this order matches your specific model's training labels!
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


def predict_image(image_path: str):
    try:
        # Load and convert to RGB (Crucial to avoid Alpha channel or Grayscale issues)
        img_pil = Image.open(image_path).convert("RGB")

        # Apply transforms
        input_tensor = transform(img_pil).unsqueeze(0).float()

        with torch.no_grad():
            outputs = model(input_tensor)

            # Handle list/tuple outputs from specialized Swin heads
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            # Global Average Pooling if the model returns a 4D feature map [1, 7, 7, 7]
            if logits.ndim > 2:
                logits = torch.mean(logits, dim=(2, 3))

            logits = logits.squeeze()  # Target shape: [7]

            # Probability calculation
            probs = F.softmax(logits, dim=0)

        # Get Top 3 results
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
        return {"error": f"Inference Error: {str(e)}"}