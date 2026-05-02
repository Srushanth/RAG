import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""<a href="https://colab.research.google.com/github/Srushanth/RAG/blob/modular-rag/Advanced-RAG/notebooks/modular-rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""),
    
    nbf.v4.new_markdown_cell("""# 🧩 Modular RAG with LlamaIndex QueryPipeline

This notebook explores **Modular RAG**. Instead of relying on rigid, high-level abstractions like `index.as_query_engine()`, we break down our RAG architecture into explicit, swappable modules:

1.  **Ingestion & Indexing Module:** Document loading, granular chunking, and embedding.
2.  **Retrieval Module:** Retrieving context nodes from the vector store.
3.  **Synthesis/Generation Module:** Structuring prompts and interacting with the LLM.

We will orchestrate these custom components using LlamaIndex's declarative `QueryPipeline`."""),
    
    nbf.v4.new_markdown_cell("""---
## 1. Setup & Configuration"""),
    
    nbf.v4.new_code_cell("""! pip install "llama-index-core>=0.14.21" "llama-index-embeddings-huggingface>=0.7.0" "llama-index-llms-google-genai>=0.9.1" "sentence-transformers>=4.0.0\""""),
    
    nbf.v4.new_code_cell("""import os
import nest_asyncio

nest_asyncio.apply()"""),
    
    nbf.v4.new_code_cell("""from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set your Gemini API Key
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY\""""),
    
    nbf.v4.new_code_cell("""# We use Gemini 1.5 Flash for rapid synthesis and a local HuggingFace embedding model
Settings.llm = GoogleGenAI(model="gemini-1.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

print(f"LLM: {Settings.llm.model}")
print(f"Embed model: {Settings.embed_model.model_name}")"""),

    nbf.v4.new_markdown_cell("""---
## 2. The Ingestion & Indexing Module

First, we explicitly define our document parsing and node extraction. Instead of a single call to build the index, we separate the parsing logic from the vector store construction. This allows us to apply custom node parsers or metadata extractors if needed."""),

    nbf.v4.new_code_cell("""from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

# Explicitly define our chunking strategy
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

print("Extracting nodes...")
nodes = splitter.get_nodes_from_documents(documents)

print(f"Extracted {len(nodes)} nodes. Building index...")
index = VectorStoreIndex(nodes)"""),

    nbf.v4.new_markdown_cell("""---
## 3. Defining the Retrieval and Synthesis Modules

Now, we define our individual RAG components.
*   **Retriever:** Finds the most similar nodes to our query.
*   **Prompt Template:** The instructional template for the LLM.
*   **Formatter:** A simple utility function to stringify our retrieved nodes so they can be injected into the prompt."""),

    nbf.v4.new_code_cell("""from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import FnComponent

# 1. Retrieval Module
retriever = index.as_retriever(similarity_top_k=3)

# 2. Synthesis Module (Prompt Template)
prompt_str = (
    "Context information is below.\\n"
    "---------------------\\n"
    "{context_str}\\n"
    "---------------------\\n"
    "Given the context information and not prior knowledge, answer the user's query.\\n"
    "Query: {query_str}\\n"
    "Answer: "
)
prompt_tmpl = PromptTemplate(prompt_str)

# 3. Utility Module: Format nodes into a single string for the prompt
def format_nodes(nodes):
    return "\\n\\n".join([n.get_content() for n in nodes])

format_nodes_c = FnComponent(fn=format_nodes)"""),

    nbf.v4.new_markdown_cell("""---
## 4. Orchestration with QueryPipeline

This is where the modularity shines. We connect our distinct components together into a logical sequence (a pipeline). If we ever want to swap the retriever, add a reranker, or route queries, we simply modify the links in this pipeline without rewriting the entire RAG application."""),

    nbf.v4.new_code_cell("""from llama_index.core.query_pipeline import QueryPipeline

# Initialize the pipeline
p = QueryPipeline(verbose=True)

# Add all our modules to the pipeline
p.add_modules({
    "retriever": retriever,
    "format_nodes": format_nodes_c,
    "prompt_tmpl": prompt_tmpl,
    "llm": Settings.llm
})

# Define the data flow (Links)
# 1. Retriever outputs nodes -> goes into format_nodes
p.add_link("retriever", "format_nodes")

# 2. format_nodes outputs string -> goes into prompt_tmpl's context_str variable
p.add_link("format_nodes", "prompt_tmpl", dest_key="context_str")

# 3. prompt_tmpl outputs a formatted prompt -> goes into the LLM
p.add_link("prompt_tmpl", "llm")"""),

    nbf.v4.new_markdown_cell("""---
## 5. Execution

To run the pipeline, we provide the input query. By default, any module expecting an input that isn't connected to another module's output will receive the initial `query_str`.

In our pipeline:
*   `retriever` needs a query.
*   `prompt_tmpl` needs `{query_str}`."""),

    nbf.v4.new_code_cell("""# Run the modular pipeline
query = "What is Apple's progress on their 2030 environmental goals?"

print(f"\\nExecuting Pipeline for query: '{query}'\\n")
output = p.run(query_str=query)

print("\\n--------------------------------")
print("Final Output:")
print("--------------------------------")
print(output.message.content)""")
]

with open('notebooks/modular-rag.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
