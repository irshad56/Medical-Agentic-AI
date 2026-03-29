def fusion_agent(processed_input: dict, image_result: dict = None, visualization_result: dict = None):
    """
    Bundles processed data into a single dictionary for the reasoning agent.
    """
    fused_data = {
        "query": processed_input.get("query"),
        "image_path": processed_input.get("image_path"),
        "image_analysis": None,
        "visualization": None
    }

    if image_result and image_result.get("status") == "success":
        fused_data["image_analysis"] = image_result.get("data")

    if visualization_result and visualization_result.get("status") == "success":
        fused_data["visualization"] = visualization_result.get("chart_path")

    return fused_data