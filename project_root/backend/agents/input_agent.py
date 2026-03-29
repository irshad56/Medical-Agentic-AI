def input_processing_agent(query: str, image_path=None):
    """
    Standardize and validate input
    """
    return {
        "query": query.strip(),
        "image_path": image_path,
        "has_image": image_path is not None
    }