import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""<a href="https://colab.research.google.com/github/Srushanth/RAG/blob/modular-rag/Modular-RAG/notebooks/modular-rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""),
    
    nbf.v4.new_markdown_cell("""# 🧩 Modular RAG with LlamaIndex Workflows

This notebook explores **Modular RAG**. Instead of relying on rigid, high-level abstractions like `index.as_query_engine()`, we break down our RAG architecture into explicit, event-driven modules.

LlamaIndex recently transitioned from `QueryPipelines` to **`Workflows`**. Workflows are built on an event-driven architecture that allows for highly customizable, scalable, and complex routing (including loops for agentic logic) that was difficult to achieve in older DAG-based pipelines.

We will build a workflow consisting of:
1.  **Ingestion & Indexing:** Document loading, chunking, and embedding.
2.  **Retrieval Step:** An event listener that fetches context from the vector store.
3.  **Synthesis Step:** An event listener that formats the prompt and calls the LLM."""),
    
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

First, we explicitly define our document parsing and node extraction. Instead of a single call to build the index, we separate the parsing logic from the vector store construction. This allows us to apply custom node parsers or metadata extractors if needed.

*Note: For this example, ensure you have a `data/` directory with a document (e.g., an Apple Environmental Progress Report).*"""),

    nbf.v4.new_code_cell("""import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Create dummy data for testing if the data directory is empty
os.makedirs("data", exist_ok=True)
if not os.listdir("data"):
    with open("data/sample.txt", "w") as f:
        f.write("Apple has committed to becoming 100% carbon neutral by 2030 across its entire supply chain and product life cycle. The company's progress is detailed in their annual Environmental Progress Report.")

print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

# Explicitly define our chunking strategy
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

print("Extracting nodes...")
nodes = splitter.get_nodes_from_documents(documents)

print(f"Extracted {len(nodes)} nodes. Building index...")
index = VectorStoreIndex(nodes)"""),

    nbf.v4.new_markdown_cell("""---
## 3. Defining the Modular RAG Workflow

In LlamaIndex, a **Workflow** is defined by creating a class that inherits from `Workflow`, and decorating asynchronous methods with `@step`.

Communication between steps is handled by **Events**.
1. `StartEvent`: A built-in event containing the user's initial input.
2. `RetrievalEvent`: A custom event we create to pass retrieved nodes from the Retriever step to the Synthesizer step.
3. `StopEvent`: A built-in event that terminates the workflow and returns the final output."""),

    nbf.v4.new_code_cell("""from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event
)
from llama_index.core import PromptTemplate

# 1. Define Custom Events
class RetrievalEvent(Event):
    \"\"\"Event containing the retrieved nodes and the original query.\"\"\"
    nodes: list
    query: str

# 2. Define the Workflow
class ModularRAGWorkflow(Workflow):
    def __init__(self, index, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.llm = llm
        
        # Define our Prompt Template module
        prompt_str = (
            "Context information is below.\\n"
            "---------------------\\n"
            "{context_str}\\n"
            "---------------------\\n"
            "Given the context information, answer the user's query.\\n"
            "Query: {query_str}\\n"
            "Answer: "
        )
        self.prompt_tmpl = PromptTemplate(prompt_str)

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievalEvent:
        \"\"\"Module 1: The Retrieval Step. Listens for StartEvent.\"\"\"
        query = ev.query
        print(f"[Step 1: Retriever] Fetching context for: '{query}'")
        
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = await retriever.aretrieve(query)
        
        # Emit a RetrievalEvent to trigger the next step
        return RetrievalEvent(nodes=nodes, query=query)

    @step
    async def synthesize(self, ev: RetrievalEvent) -> StopEvent:
        \"\"\"Module 2: The Synthesis Step. Listens for RetrievalEvent.\"\"\"
        print(f"[Step 2: Synthesizer] Formatting prompt and calling LLM...")
        
        # Format the retrieved nodes into a single string
        context_str = "\\n\\n".join([n.get_content() for n in ev.nodes])
        
        # Format the prompt using our template
        formatted_prompt = self.prompt_tmpl.format(
            context_str=context_str,
            query_str=ev.query
        )
        
        # Call the LLM Module
        response = await self.llm.acomplete(formatted_prompt)
        
        # Return a StopEvent with the final result
        return StopEvent(result=str(response))"""),

    nbf.v4.new_markdown_cell("""---
## 4. Execution

To run the workflow, we instantiate our custom class and call the async `.run()` method, passing the initial query."""),

    nbf.v4.new_code_cell("""# Instantiate the modular workflow
workflow = ModularRAGWorkflow(index=index, llm=Settings.llm)

query = "What is Apple's progress on their 2030 environmental goals?"

print(f"\\nExecuting Workflow for query: '{query}'\\n")

# Run the workflow (we must await it since workflows are natively asynchronous)
result = await workflow.run(query=query)

print("\\n--------------------------------")
print("Final Output:")
print("--------------------------------")
print(result)""")
]

with open('notebooks/modular-rag.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
