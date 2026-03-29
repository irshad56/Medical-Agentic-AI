# 🩺 Medical Agentic AI v2.0
**Advanced Clinical Decision Support powered by Swin-T & Llama-3-70B**

Medical Agentic OS is a sophisticated AI-driven dermatological analysis platform. It leverages a custom-trained **Swin Transformer** for high-precision image classification and a **Llama-3-70B Agentic Workflow** (via LangGraph) to provide clinical reasoning, visualization, and patient-centric reports.

---

## 🚀 Key Features
* **Precision Vision:** Custom Swin Transformer model fine-tuned on the ISIC dataset for skin lesion classification.
* **Agentic Orchestration:** Multi-agent system (Input, Image, Visualization, and Reasoning agents) orchestrated via LangGraph.
* **Interactive UI:** Modern, professional Streamlit interface designed for clinical workflows.
* **Automated Visualization:** Dynamic generation of confidence scores and clinical charts.
* **Secure & Scalable:** Built with Python, Groq API for ultra-fast inference, and Git LFS for model management.

---

## 📂 Project Structure
```text
Agentic_AI/
├── project_root/
│   ├── backend/          # LangGraph agents & logic
│   ├── frontend/         # Streamlit UI implementation
│   ├── models/           # Swin-T model weights (.pth)
│   ├── data/             # Image cache and metadata
│   └── requirements.txt  # Dependencies
└── main.py               # Entry point