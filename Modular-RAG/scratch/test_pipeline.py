import os
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set your Gemini API Key
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "dummy_key_for_setup"

Settings.llm = GoogleGenAI(model="gemini-1.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

print("Loading documents...")
# Just load a small piece to test
documents = SimpleDirectoryReader("data").load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents[:1]) # Only one document for quick test

print("Building index...")
index = VectorStoreIndex(nodes)

retriever = index.as_retriever(similarity_top_k=2)

from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline, FnComponent

prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
prompt_tmpl = PromptTemplate(prompt_str)

def format_nodes(nodes):
    return "\n\n".join([n.get_content() for n in nodes])

format_nodes_c = FnComponent(fn=format_nodes)

print("Building pipeline...")
p = QueryPipeline(verbose=True)
p.add_modules({
    "retriever": retriever,
    "format_nodes": format_nodes_c,
    "prompt_tmpl": prompt_tmpl,
    "llm": Settings.llm
})

p.add_link("retriever", "format_nodes")
p.add_link("format_nodes", "prompt_tmpl", dest_key="context_str")
p.add_link("prompt_tmpl", "llm")

print("Pipeline built successfully!")
