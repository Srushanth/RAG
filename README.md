# Retrieval-Augmented Generation (RAG) Demo

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using a domain-specific PDF document as a knowledge source. It showcases how to combine information retrieval and generative AI to answer questions based on external documents.

## 📄 Data Source

The RAG system references the following industry report:

**EY India Energy Sector at Cross Roads**  
[Download PDF](https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/energy-resources/documents/2024/ey-india-energy-sector-at-cross-roads.pdf)

This report provides critical insights into trends, challenges, and opportunities within India's energy landscape.

## 🚀 Features

- PDF ingestion and chunking for semantic search
- Embedding generation for document chunks using HuggingFace BAAI/bge-small-en-v1.5
- Retrieval of contextually relevant sections using vector search
- Question-answer generation using Ollama with Gemma3:1b model
- Colorful console logging for better debugging experience
- Interactive command-line interface

## 🧰 Tech Stack

- **Python 3.8+**
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv)
- **LLM Framework**: LlamaIndex
- **Local LLM**: Ollama (Gemma3:1b)
- **Embeddings**: HuggingFace (BAAI/bge-small-en-v1.5)
- **Logging**: colorlog for enhanced console output

## 📋 Prerequisites

### 1. Install uv Package Manager

#### Windows (uv)

```powershell
# Using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS (uv)

```bash
# Using Homebrew
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Linux (uv)

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Install Ollama

#### Windows (Ollama)

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Open Command Prompt/PowerShell and run:

```powershell
ollama pull gemma2:2b
```

#### macOS (Ollama)

```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com/download
# Then pull the model
ollama pull gemma2:2b
```

#### Linux (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull gemma2:2b
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create Virtual Environment with uv

#### Windows (PowerShell/Command Prompt) (venv)

```powershell
# Create virtual environment
uv venv

# Activate virtual environment
# For PowerShell:
.venv\Scripts\activate.ps1
# For Command Prompt:
.venv\Scripts\activate.bat
```

#### macOS/Linux (venv)

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install project dependencies
uv pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface colorlog
```

### 4. Download the PDF Document

1. **Create the data directory:**

   ```bash
   mkdir data
   ```

2. **Download the PDF:**
   - Visit: [EY India Energy Sector Report](https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/energy-resources/documents/2024/ey-india-energy-sector-at-cross-roads.pdf)
   - Save the PDF file as `ey-india-energy-sector-at-cross-roads.pdf` in the `data/` directory

   **Alternative using curl/wget:**

   #### Windows (PowerShell)

   ```powershell
   Invoke-WebRequest -Uri "https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/energy-resources/documents/2024/ey-india-energy-sector-at-cross-roads.pdf" -OutFile "data/ey-india-energy-sector-at-cross-roads.pdf"
   ```

   #### macOS/Linux

   ```bash
   curl -o "data/ey-india-energy-sector-at-cross-roads.pdf" "https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/energy-resources/documents/2024/ey-india-energy-sector-at-cross-roads.pdf"
   ```

### 5. Start Ollama Service

#### Windows

```powershell
# Ollama should start automatically after installation
# If not, run:
ollama serve
```

#### macOS

```bash
# Start Ollama service
ollama serve
```

#### Linux

```bash
# Start Ollama service
ollama serve
```

### 6. Run the Application

```bash
python src/main.py
```

## 💻 Usage

Once the application starts, you'll see colorful logs indicating the initialization process:

```bash
2025-06-21 07:00:43,038 - INFO - Setting up Ollama LLM with gemma3:1b model
2025-06-21 07:00:43,040 - INFO - Loading embeddings...
2025-06-21 07:00:47,246 - INFO - Loading documents from data directory...
2025-06-21 07:00:48,297 - INFO - Loaded 25 documents
2025-06-21 07:00:48,298 - INFO - Creating vector store index...
2025-06-21 07:00:56,494 - INFO - Application initialized successfully. Ready for queries.
```

### Example Queries

```bash
Enter your query: What are the key challenges in India's energy sector?
Enter your query: What renewable energy trends are mentioned in the report?
Enter your query: What are the policy recommendations for energy transition?
Enter your query: exit
```

## 🗂️ Project Structure

```bash
project-root/
├── src/
│   └── main.py          # Main application entry point
├── data/
│   └── ey-india-energy-sector-at-cross-roads.pdf  # PDF document
├── .venv/               # Virtual environment (created by uv)
├── application.log      # Application logs (created at runtime)
└── README.md           # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama not running:**

   ```bash
   # Check if Ollama is running
   ollama list

   # If not running, start it
   ollama serve
   ```

2. **Model not found:**

   ```bash
   # Pull the required model
   ollama pull gemma2:2b
   ```

3. **Virtual environment issues:**

   ```bash
   # Deactivate and recreate
   deactivate
   rm -rf .venv  # or rmdir /s .venv on Windows
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

4. **PDF not found:**
   - Ensure the PDF is downloaded and placed in the `data/` directory
   - Check the filename matches exactly

5. **Dependencies issues:**

   ```bash
   # Reinstall dependencies
   uv pip install --upgrade llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface colorlog
   ```

## 🔧 Configuration

### Changing the LLM Model

Edit `src/main.py` and modify:

```python
Settings.llm = Ollama(model="your-preferred-model", request_timeout=60.0)
```

### Changing the Embedding Model

Edit the `load_embeddings()` function in `src/main.py`:

```python
return HuggingFaceEmbedding(model_name="your-preferred-embedding-model")
```

## 📝 Logging

The application uses colorful logging with the following color scheme:

- 🟢 **Green**: INFO messages
- 🟡 **Yellow**: WARNING messages
- 🔴 **Red**: ERROR messages
- 🔴 **Red with white background**: CRITICAL messages

Logs are displayed in the console with colors and saved to `application.log` without colors.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📬 Questions?

Feel free to reach out or open an issue!

## 📄 License

This project is for research and demonstration purposes only. The PDF and content remain the property of Ernst & Young.

---

**Note**: Make sure to keep your virtual environment activated when working with the project. You can deactivate it using the `deactivate` command when you're done.
