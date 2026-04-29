---
title: "Building a Naive RAG Application from Scratch with LlamaIndex and Gemini"
date: "2026-04-22"
tags: ["AI", "RAG", "LlamaIndex", "Streamlit", "LLM", "Python", "Gemini"]
description: "A step-by-step journey of building a local document chatbot, from a Jupyter Notebook proof-of-concept to a fully interactive Streamlit web application."
---

# Building a Naive RAG Application from Scratch with LlamaIndex and Gemini

![Building a Naive RAG Application from Scratch with LlamaIndex and Gemini](https://storage.googleapis.com/portfolio-srushanth-baride-images/Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini/landing-image.png)

Retrieval-Augmented Generation (RAG) is the backbone of modern enterprise AI. But before diving into complex, multi-agent RAG architectures with query rewriting and semantic routing, it's critical to understand the fundamentals.

Today, I want to walk through how I built a "Naive RAG" application the simplest, purest form of RAG using LlamaIndex, Google's Gemini, HuggingFace embeddings, and Streamlit.

![High Level Naive RAG Architecture](https://storage.googleapis.com/portfolio-srushanth-baride-images/Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini/High-Level-Naive-RAG-Architecture.png)

## The Tech Stack

To keep things lightweight but powerful, I chose:

- **Data Framework:** [LlamaIndex](https://www.llamaindex.ai/)
- **LLM:** Google Gemini (`gemini-1.5-flash`)
- **Embeddings:** HuggingFace local models (`BAAI/bge-small-en-v1.5`)
- **UI:** [Streamlit](https://streamlit.io/)

## Phase 1: The Notebook Proof of Concept

I started in a Jupyter Notebook. The goal was simple: point LlamaIndex at a `data/` directory, embed the documents, and ask questions.

The code was elegantly simple:

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Setup models
Settings.llm = GoogleGenAI(model="gemini-1.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")

# Load and index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine(similarity_top_k=3)
print(query_engine.query("What is the main topic of these documents?"))
```

### The Rate Limit Reality Check

Everything worked perfectly until I threw a **22 MB Apple Environmental Progress Report** at it.

Because I was using the free tier of the Gemini API, I immediately hit the `429 Too Many Requests` quota. LlamaIndex batches documents for embedding, and processing thousands of chunks instantly overwhelmed the 15 Requests-Per-Minute limit of the free tier.

**The Solution:** Hybrid architecture. I kept Gemini for the LLM inference (where rate limits are less heavily punished for single queries) but switched the embedding model to run _locally_ using HuggingFace's `bge-small-en-v1.5`.

```bash
uv add llama-index-embeddings-huggingface torch torchvision
```

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

By generating embeddings locally, I bypassed the API rate limits entirely and retained the blazing-fast response time of Gemini for the final text generation.

![Hybrid Embedding Workflow](https://storage.googleapis.com/portfolio-srushanth-baride-images/Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini/Hybrid-Embedding-Workflow.png)

### The Performance Reality Check: CPU vs. GPU

While local embeddings solved the API rate limits, they introduced a new bottleneck: compute power.

When I first ran the indexing step:

```python
index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)
```

on a standard CPU, processing and embedding the document chunks took a grueling **45 minutes**.

To speed things up, I switched my environment to utilize an NVIDIA T4 GPU. With hardware acceleration enabled, that exact same indexing step dropped to just **6 minutes**. If you are building a local RAG pipeline with a sizable dataset, running your embedding model on a GPU is practically mandatory for a good developer experience.

![CPU Vs GPU Performance](https://storage.googleapis.com/portfolio-srushanth-baride-images/Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini/CPU-Vs-GPU-Performance.png)

---

## Phase 2: Escaping the Notebook with Streamlit

A notebook is great for testing, but a chatbot needs a UI. I decided to wrap the RAG engine in a Streamlit app.

### The `asyncio` Trap

Moving LlamaIndex out of a notebook and into a Python script often triggers a notorious error:
`RuntimeError: asyncio.run() cannot be called from a running event loop`

This happens because modern AI SDKs often try to spin up new asynchronous event loops under the hood. The fix is remarkably simple just patch the event loop before importing the AI libraries:

```python
import nest_asyncio
nest_asyncio.apply()
```

### The Final Application

With the asynchronous bugs squashed, the final Streamlit app came together beautifully. By using Streamlit's `@st.cache_resource`, I ensured the heavy lifting loading and embedding the documents only happened once when the app launched, rather than every time the user sent a message.

![Streamlit UI Mockup](https://storage.googleapis.com/portfolio-srushanth-baride-images/Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini/Streamlit-UI-Mockup.png)

I also added a secure sidebar input for the Gemini API key, ensuring the code remained safe to share publicly without exposing credentials.

```python
# Caching the index so it doesn't rebuild on every chat message
@st.cache_resource(show_spinner=False)
def initialize_engine():
    # Load settings & documents
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(similarity_top_k=3)
```

## Conclusion

Building a Naive RAG pipeline is incredibly empowering. While production systems might require vector databases like Pinecone, rerankers, and complex retrieval strategies, this simple in-memory architecture is more than capable of handling personal document chat.

By understanding how the pieces fit together and how to navigate API rate limits by strategically mixing cloud LLMs with local embeddings you set the foundation for building truly robust AI applications.
