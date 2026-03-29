import streamlit as st
import os
import sys
from pathlib import Path

# ==========================================================
# ✅ FIXED PATHING: Point to the Project Root (Agentic_AI)
# ==========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# Now Python can find 'backend' inside the project root
from backend.graph.langgraph_flow import run_pipeline

# Page Configuration
st.set_page_config(
    page_title="Medical Agentic OS v2.0",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Medical/Tech" feel
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stStatusWidget { border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_viz" not in st.session_state:
    st.session_state.last_viz = None
if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None

st.title("🩺 Medical Agentic OS v2.0")
st.caption("Powered by Swin Transformer & Llama-3 Reasoning Agents")

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("🕵️ Control Panel")

    # 1. Image Upload
    uploaded_file = st.file_uploader(
        "Upload Skin Lesion Image (JPG/PNG)",
        type=["jpg", "png", "jpeg"],
        help="Upload a photo for Swin Transformer analysis",
        key="uploader_main"
    )

    # Logic to handle new uploads and clear old session logic if necessary
    if uploaded_file:
        u_dir = BASE_DIR / "data" / "images"
        os.makedirs(u_dir, exist_ok=True)

        # Save file
        temp_path = str(u_dir / uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Update session state if the image is actually different
        if st.session_state.current_image_path != temp_path:
            st.session_state.current_image_path = temp_path
            # Optional: Clear last viz if you want a fresh start per image
            st.session_state.last_viz = None

        st.success(f"Image Cached: {uploaded_file.name}")
        st.image(st.session_state.current_image_path, caption="Current Analysis Subject", use_container_width=True)

        with st.expander("File Metadata"):
            st.text(f"Name: {uploaded_file.name}")
            st.text(f"Path: {st.session_state.current_image_path}")
    else:
        st.session_state.current_image_path = None

    st.divider()

    if st.button("🗑️ Clear Memory & Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_viz = None
        st.session_state.current_image_path = None
        st.rerun()

# --- CHAT INTERFACE ---

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg and msg["image"]:
            st.image(msg["image"], caption="Generated Analysis Chart")

# Handle User Input
if query := st.chat_input("Ask about the image or request a visualization..."):
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        status_box = st.status("🧬 Orchestrating Agents...", expanded=True)
        log_placeholder = st.empty()
        live_logs = []


        def update_ui_logs(msg):
            live_logs.append(msg)
            log_placeholder.markdown("\n".join([f"{m}" for m in live_logs[-5:]]))


        try:
            # 🚀 Calling the Orchestrator pipeline with the STABLE session path
            response_text, viz_path = run_pipeline(
                query=query,
                image_path=st.session_state.current_image_path,
                log_callback=update_ui_logs
            )

            status_box.update(label="✅ Medical Analysis Complete", state="complete", expanded=False)
            st.markdown(response_text)

            history_entry = {"role": "assistant", "content": response_text, "image": None}

            if viz_path and os.path.exists(viz_path):
                st.image(viz_path, caption="Agentic Visualization Output")
                history_entry["image"] = viz_path
                st.session_state.last_viz = viz_path

            st.session_state.chat_history.append(history_entry)

        except Exception as e:
            status_box.update(label="❌ System Error", state="error")
            st.error(f"An error occurred: {str(e)}")