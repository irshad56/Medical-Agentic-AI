import os
from pathlib import Path
from backend.graph.langgraph_flow import run_pipeline

# 1. Get the absolute path to the 'project_root'
# Since we are in project_root/backend/agents, we go up 2 levels
BASE_DIR = Path(__file__).resolve().parents[2]

# 2. Join it with the data folder
# This creates: C:\...\project_root\data\images\brain_Tumor Image(Gelioma).jpg
image_path = str(BASE_DIR / "data" / "images" / "brain_Tumor Image(Gelioma).jpg")

if __name__ == "__main__":
    query = "Analyze this brain image and explain results"

    # Run the pipeline with the absolute path
    result = run_pipeline(query, image_path)
    print("\n--- FINAL RESPONSE ---")
    print(result)