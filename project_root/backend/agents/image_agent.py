import os
import sys
from pathlib import Path

# Ensure the project root is in sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from backend.tools.ml_model import predict_image


def image_understanding_agent(image_path: str):
    absolute_path = os.path.abspath(image_path)

    if not os.path.exists(absolute_path):
        return {
            "status": "failed",
            "data": {"error": f"File not found at {absolute_path}"}
        }

    result = predict_image(absolute_path)

    if "error" in result:
        return {
            "status": "failed",
            "data": result
        }

    return {
        "status": "success",
        "data": result  # now contains top_3
    }