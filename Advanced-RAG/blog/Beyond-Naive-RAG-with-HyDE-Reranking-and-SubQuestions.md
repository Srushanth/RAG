---
title: "Beyond Naive RAG: Leveling up with HyDE, Re-ranking, and Sub-Questions in LlamaIndex"
date: "2026-04-29"
tags: ["AI", "RAG", "LlamaIndex", "LLM", "Python", "Gemini", "Machine Learning"]
description: "A deep dive into three advanced Retrieval-Augmented Generation techniques: HyDE, Cross-Encoder Re-ranking, and Sub-Question query decomposition."
---

# Beyond Naive RAG: Leveling up with HyDE, Re-ranking, and Sub-Questions

In a [previous post](Building-a-Naive-RAG-Application-from-Scratch-with-LlamaIndex-and-Gemini.md), I walked through the process of building a foundational "Naive RAG" application from scratch using LlamaIndex and Google's Gemini. That pipeline was simple and effective: load documents, embed them into a vector database, and perform a basic similarity search to retrieve context for the LLM. 

However, as you scale up your documents or face more complex user queries, Naive RAG starts to show its limitations:
- **Vocabulary Mismatch:** The words in the user's short query might not match the terminology used in the dense documents.
- **Lost in the Middle:** Retrieving too many chunks introduces noise, but retrieving too few risks missing the answer.
- **Complex Questions:** Simple similarity search fails when a question requires aggregating information from multiple distinct parts of the document.

To solve these issues, I set up a new lab environment to experiment with three **Advanced RAG** techniques: **HyDE**, **Re-ranking**, and the **Sub-Question Engine**.

---

## The Baseline Setup

To establish a baseline, I used the same hybrid architecture as before: using a robust local embedding model (`BAAI/bge-small-en-v1.5`) via HuggingFace to avoid rate limits, while relying on Gemini (`gemini-1.5-flash`) for the heavy-lifting generation.

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configure the LLM and Local Embeddings
Settings.llm = GoogleGenAI(model="gemini-1.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

# 2. Build the Index and Baseline Query Engine
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
baseline_engine = index.as_query_engine(similarity_top_k=3)
```

With the baseline running, it was time to experiment.

---

## Technique 1: HyDE (Hypothetical Document Embeddings)

### The Problem
When a user asks a short, abstract question (e.g., "What is the company's stance on remote work?"), the raw vector of that question might not semantically match the dense, formal policy text inside the employee handbook.

### The HyDE Solution
HyDE flips the script: before doing any retrieval, it asks the LLM to write a *fake, hypothetical answer* to the user's question based entirely on its pre-trained knowledge. Even if this fake answer contains hallucinations or is factually wrong, its *semantic structure and vocabulary* will closely mirror the actual document we are trying to find! 

We then embed this hypothetical answer and use *that* to search the vector database.

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# Create the HyDE transform
hyde = HyDEQueryTransform(include_original=True)

# Wrap the baseline engine
hyde_query_engine = TransformQueryEngine(baseline_engine, query_transform=hyde)

response = hyde_query_engine.query("What is the company's stance on remote work?")
```

**Verdict:** HyDE drastically improves retrieval for short, ambiguous queries. The trade-off is an extra LLM call (to generate the fake answer), which adds a slight latency overhead.

---

## Technique 2: Re-ranking with Cross-Encoders

### The Problem
Standard vector search uses "bi-encoders" which pre-calculate vectors for documents and queries independently and compare them via dot-product. It's incredibly fast, but misses subtle semantic nuances. If we retrieve 10 documents, the #1 result isn't always the most relevant.

### The Re-ranking Solution
Instead of relying solely on the initial retrieval, we can use a **Cross-Encoder**. A cross-encoder takes the query and a document chunk simultaneously and scores their relevance together. It's too slow to run across your entire database, but perfect for re-scoring the top-10 results from a fast vector search.

I used `BAAI/bge-reranker-v2-m3`, a powerful local reranker that runs without an API key.

```python
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# Initialize the local cross-encoder
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=3)

# Retrieve 10 chunks, but only keep the top 3 after re-ranking
rerank_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)
```

**Verdict:** Re-ranking is arguably the highest ROI optimization you can add to a RAG pipeline. It allows you to cast a wide net (high recall) during the initial vector search, and uses the cross-encoder to guarantee high precision in the final context window.

---

## Technique 3: The Sub-Question Query Engine

### The Problem
If a user asks, *"Compare the Q1 revenue of Apple to the Q1 revenue of Microsoft,"* standard RAG fails. A single similarity search cannot pull completely disconnected financial data for two different companies reliably.

### The Sub-Question Solution
The Sub-Question Engine intercepts the complex query and uses the LLM to decompose it into independent, actionable sub-questions. 
1. *What was Apple's Q1 revenue?*
2. *What was Microsoft's Q1 revenue?*

It then executes these sub-queries in parallel against your tools, gathers the distinct answers, and synthesizes a final comparative response.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Create a tool from our base engine
query_engine_tool = QueryEngineTool(
    query_engine=baseline_engine,
    metadata=ToolMetadata(
        name="company_financials",
        description="Provides information about company financial reports."
    ),
)

# Initialize the Sub-Question Engine
question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[query_engine_tool],
    question_gen=question_gen
)
```

**Verdict:** This is where RAG starts to feel like an agent. While it is the slowest technique due to the multiple synthesis steps and LLM calls, it is absolutely essential for analytical, multi-hop reasoning tasks.

---

## Summary & Trade-offs

Each technique serves a specific purpose in a production RAG pipeline:

| Technique | Best For | Trade-off |
|-----------|----------|----------|
| **Baseline** | Simple, well-formed queries | Fast, but struggles with noise and complex logic |
| **HyDE** | Short/abstract queries, vocabulary mismatch | Slight latency bump (Extra LLM call) |
| **Re-ranking** | Improving precision from noisy retrieval | Extra model inference overhead |
| **Sub-Question** | Complex, multi-part, or comparative questions | Slowest (Multiple LLM calls & retrieval passes) |

Building advanced RAG isn't about applying every technique simultaneously—it's about understanding the specific failure modes of your naive architecture and applying the right tool for the job. 

Whether it's bridging a vocabulary gap with HyDE, boosting precision with a Re-ranker, or breaking down analytical problems with Sub-Questions, LlamaIndex makes it incredibly straightforward to evolve your pipeline.
