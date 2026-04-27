# 🚀 Advanced RAG with LlamaIndex

> Enhance retrieval quality through **pre-retrieval query transformation** and **post-retrieval optimization** — three experiments you can toggle independently.

## What is Advanced RAG?

**Naive RAG** follows a simple *retrieve → generate* pattern. **Advanced RAG** adds optimisation stages **before** and **after** retrieval to improve the quality and relevance of the context fed to the LLM.

```
┌───────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  User Query   │────▶│ Pre-Retrieval│────▶│   Retrieval   │────▶│Post-Retrieval│────▶ Response
│               │     │  (HyDE /     │     │  (Vector      │     │  (Re-ranking)│
│               │     │   SubQ)      │     │   Search)     │     │              │
└───────────────┘     └──────────────┘     └───────────────┘     └──────────────┘
```

## Techniques

| # | Technique | Stage | How It Works |
|---|-----------|-------|-------------|
| 1 | **HyDE** (Hypothetical Document Embeddings) | Pre-retrieval | The LLM generates a *hypothetical answer* to the query. That answer is embedded and used for vector search instead of the raw query — yielding better semantic matches. |
| 2 | **Re-ranking** (Cross-Encoder) | Post-retrieval | Retrieves top-K candidates via vector search, then a cross-encoder model (`BAAI/bge-reranker-v2-m3`) rescores each (query, chunk) pair. The top-N highest-scoring chunks are kept. |
| 3 | **Sub-Question Engine** | Pre-retrieval | Complex multi-part questions are decomposed into independent sub-questions. Each sub-question is answered separately, then the results are synthesised into one coherent response. |

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### Setup

```bash
# Clone the repo and cd into the project
cd Advanced-RAG

# Install dependencies
uv sync

# Place your documents in the data/ folder
cp /path/to/your/docs/* data/

# Launch the app
uv run streamlit run app.py
```

### Usage

1. Enter your **Gemini API key** in the sidebar.
2. Select a **technique** (Baseline / HyDE / Re-ranking / Sub-Question).
3. Adjust retrieval parameters (top-K, top-N) if desired.
4. Ask questions about your documents!

## Project Structure

```
Advanced-RAG/
├── app.py                  # Streamlit application
├── data/                   # Place your documents here
├── notebooks/
│   └── advanced-rag.ipynb  # Experimentation notebook
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `llama-index-core` | Core RAG framework |
| `llama-index-llms-google-genai` | Gemini LLM integration |
| `llama-index-embeddings-huggingface` | Local embedding model |
| `llama-index-postprocessor-sbert-rerank` | Local cross-encoder re-ranker |
| `sentence-transformers` | Underlies the SBERT re-ranker |
| `streamlit` | Web UI |

## References

- [LlamaIndex — HyDE Query Transform](https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/)
- [LlamaIndex — Node Postprocessors (Re-ranking)](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
- [LlamaIndex — Sub-Question Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
