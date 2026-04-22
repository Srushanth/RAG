# Naive Simple RAG

A clean, minimal, and fully functional Retrieval-Augmented Generation (RAG) application built with Python, LlamaIndex, Google Gemini, and Streamlit.

This project demonstrates how to index your own local documents and chat with them using a modern, interactive web UI.

## Architecture
- **Framework**: [LlamaIndex](https://www.llamaindex.ai/) (Data framework for LLM applications)
- **UI**: [Streamlit](https://streamlit.io/) (Interactive Chat Interface)
- **LLM**: Google Gemini (`gemini-3-flash-preview` via `google-genai`)
- **Embeddings**: HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Vector Store**: In-Memory LlamaIndex store

## Getting Started

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed to manage your environment and dependencies.

### 2. Add Your Documents
Ensure you have a `data/` directory in the root of the project and place your files (e.g., PDFs, text files) inside it.

### 3. Install Dependencies
Install all required dependencies defined in the `pyproject.toml` (if you haven't already):
```bash
uv sync
```

*Note: Since you are running HuggingFace embeddings locally, you may need `torch` and `torchvision` installed.*

### 4. Run the Application
Launch the Streamlit web server:
```bash
uv run streamlit run app.py
```

### 5. Configuration
Once the web UI opens in your browser:
1. Enter your **Google Gemini API Key** securely into the sidebar.
2. The app will automatically read, index, and embed any documents found in the `data/` directory.
3. Start chatting with your data!
