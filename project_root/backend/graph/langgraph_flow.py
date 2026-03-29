import time
import os
from backend.agents.query_agent import query_understanding_agent
from backend.agents.image_agent import image_understanding_agent
from backend.agents.visualization_agent import visualization_agent
from backend.agents.memory_agent import memory_store
from backend.agents.fusion_agent import fusion_agent
from backend.agents.reasoning_agent import reasoning_agent


def run_pipeline(query: str, image_path=None, log_callback=None):
    def log(msg, icon="⚙️"):
        if log_callback: log_callback(f"{icon} {msg}")
        time.sleep(0.1)

    # 1. BRAIN: Understand Intent
    log("Brain: Deciphering user intent...", "🧠")
    intent_data = query_understanding_agent(query)
    history = memory_store.get_full_context()

    # 2. DYNAMIC ML: Only run if the brain says so
    analysis_data = None
    if intent_data.get("requires_ml") and image_path:
        log("ML Agent: Running Swin Transformer Inference...", "🔍")
        res = image_understanding_agent(image_path)
        if res["status"] == "success":
            analysis_data = res["data"]
            memory_store.update_metadata("last_analysis", analysis_data)
    else:
        # Pull from memory if not a new analysis request
        analysis_data = memory_store.metadata.get("last_analysis")

    # 3. DYNAMIC VIZ: Only run if the brain says so
    viz_res = None
    if intent_data.get("requires_viz"):
        if analysis_data:
            log("Viz Agent: Generating dynamic visualization...", "📊")
            viz_res = visualization_agent({"status": "success", "data": analysis_data}, "bar_chart")
            if viz_res.get("status") == "success":
                memory_store.update_metadata("last_viz_path", viz_res.get("chart_path"))
        else:
            log("Viz Agent: No analysis data available to visualize.", "⚠️")

    # 4. REASONING: Formulate response based on what agents actually did
    log("Reasoning Agent: Formulating medical report...", "✍️")
    fused = fusion_agent(
        processed_input={"query": query, "image_path": image_path},
        image_result={"status": "success", "data": analysis_data} if analysis_data else None,
        visualization_result=viz_res
    )

    response = reasoning_agent(fused, history)

    memory_store.add_chat("user", query)
    memory_store.add_chat("assistant", response)

    # Return visualization path only if one was created this turn
    current_viz = viz_res.get("chart_path") if viz_res else None

    return response, current_viz