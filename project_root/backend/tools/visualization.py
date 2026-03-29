import matplotlib.pyplot as plt
import os
from pathlib import Path

# Get the project root (3 levels up from this file: tools -> backend -> project_root)
BASE_DIR = Path(__file__).resolve().parents[2]
SAVE_DIR = BASE_DIR / "data" / "images"

# Ensure the directory exists so we don't get FileNotFoundError
os.makedirs(SAVE_DIR, exist_ok=True)

def bar_chart(label, confidence):
    plt.figure()
    plt.bar([label], [confidence], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Confidence Score")
    plt.title("Model Prediction Confidence")

    path = str(SAVE_DIR / "bar_chart.png")
    plt.savefig(path)
    plt.close()
    return path

def confidence_gauge(label, confidence):
    plt.figure(figsize=(6, 6))
    plt.pie([confidence, 1-confidence],
            labels=["Confidence", "Uncertainty"],
            colors=["#4CAF50", "#E0E0E0"],
            autopct='%1.1f%%',
            startangle=90)

    plt.title(f"Prediction: {label}")

    path = str(SAVE_DIR / "gauge_chart.png")
    plt.savefig(path)
    plt.close()
    return path

def feature_placeholder():
    plt.figure()
    plt.text(0.5, 0.5, "Feature Importance (Heatmap Coming Soon)",
             ha='center', va='center', fontsize=12)

    path = str(SAVE_DIR / "feature_chart.png")
    plt.savefig(path)
    plt.close()
    return path