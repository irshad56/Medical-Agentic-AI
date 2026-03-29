import matplotlib.pyplot as plt
import os
from pathlib import Path
from backend.tools.visualization import bar_chart, confidence_gauge

# Project root setup
BASE_DIR = Path(__file__).resolve().parents[2]
SAVE_DIR = BASE_DIR / "data" / "images"
os.makedirs(SAVE_DIR, exist_ok=True)


def multi_class_bar_chart(top3):
    """Generates a bar chart for the top 3 predictions."""
    labels = [item["label"] for item in top3]
    values = [item["confidence"] for item in top3]

    save_path = str(SAVE_DIR / "top3_chart.png")

    plt.figure(figsize=(7, 5))
    colors = ['#2196F3', '#64B5F6', '#BBDEFB']  # Shading by confidence
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 1)
    plt.title("Top-3 Disease Probabilities", fontsize=14, pad=15)
    plt.ylabel("Confidence Level")
    plt.xticks(rotation=15)

    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02,
                 f'{yval:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def visualization_agent(image_result: dict, user_choice: str):
    """
    Main entry point for visuals.
    Prioritizes top_3 charts if data is available.
    """
    data = image_result.get("data", {})

    try:
        # Check if we have multi-class data
        if "top_3" in data:
            path = multi_class_bar_chart(data["top_3"])
        else:
            # Fallback to single label logic
            label = data.get("label", "Unknown")
            val = data.get("confidence", 0.5)

            if user_choice == "confidence_gauge":
                path = confidence_gauge(label, val)
            else:
                path = bar_chart(label, val)

        return {"status": "success", "chart_path": path}
    except Exception as e:
        return {"status": "failed", "data": str(e)}