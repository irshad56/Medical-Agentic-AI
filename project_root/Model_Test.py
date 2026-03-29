import torch
import os
import timm
from timm.models.swin_transformer import SwinTransformer

# 1. Register the SwinTransformer class as safe to unpickle
torch.serialization.add_safe_globals([SwinTransformer])

# The path to your complete model file
file_path = r"C:\Users\irfan\Downloads\irshad bin\Agentic_AI\project_root\models\image_model\swin_isic_phase2_full_model.pth"


def verify_and_load():
    if not os.path.exists(file_path):
        print(f"❌ File not found! Please check: {file_path}")
        return

    try:
        print(f"🔄 Loading full Medical AI model from: {file_path}")

        # 2. Set weights_only=False because we are loading a full model object, not just weights
        model = torch.load(
            file_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )

        model.eval()
        print("✅ SUCCESS: Full model and architecture loaded!")
        print(f"Model Architecture: {type(model)}")

        if hasattr(model, 'head'):
            # This will likely show the 'fc' layer structure from your fine-tuning
            print(f"Verified Model Head: {model.head}")

    except Exception as e:
        print(f"❌ Loading Error: {e}")


if __name__ == "__main__":
    verify_and_load()