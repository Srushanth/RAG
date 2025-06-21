"""
Main Module: main.py

Author: Srushanth Baride
Email: Srushanth.Baride@gmail.com
Date: 2025-06-17

Summary:
This script serves as the entry point for the project. It initializes key components,
executes core logic, and manages application flow.

Usage:
Run this script using:
    python src/main.py

Functions:
- main(): Initializes and runs the program.

Notes:
Ensure all dependencies are installed before execution.
"""

import logging
from typing import List
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama  # type: ignore
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.core.base.base_query_engine import BaseQueryEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_embeddings() -> HuggingFaceEmbedding:
    """Load and return the HuggingFace embeddings."""
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def main() -> None:
    """Main function to execute the program."""
    print("Initializing the application...")
    Settings.llm = Ollama(model="gemma3:1b", request_timeout=60.0)
    print("Loading embeddings...")
    Settings.embed_model = load_embeddings()
    print("Loading documents...")
    documents: List[Document] = SimpleDirectoryReader(input_dir="data").load_data()
    index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)
    query_engine: BaseQueryEngine = index.as_query_engine()
    while True:
        query: str = input("Enter your query: ")
        if query.lower() == "exit":
            break
        response = query_engine.query(query)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
