import os
import nest_asyncio

# Apply nest_asyncio to prevent event loop issues between Streamlit and LlamaIndex
nest_asyncio.apply()

import streamlit as st

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import TransformQueryEngine, SubQuestionQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Advanced RAG", page_icon="🚀", layout="centered")
st.title("🚀 Advanced RAG — Experiment Lab")
st.caption("Compare retrieval techniques: Baseline · HyDE · Re-ranking · Sub-Question")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get your key from Google AI Studio",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown("---")

    # ── Technique Selector ──
    st.subheader("🧪 Experiment Mode")
    technique = st.radio(
        "Select a retrieval technique:",
        options=["Baseline", "HyDE", "Re-ranking", "Sub-Question"],
        index=0,
        help=(
            "Each technique runs as a **separate experiment**.\n\n"
            "• **Baseline** — simple vector retrieval\n"
            "• **HyDE** — LLM generates a hypothetical answer, embeds that instead\n"
            "• **Re-ranking** — retrieves top-K, re-ranks with a cross-encoder\n"
            "• **Sub-Question** — decomposes complex queries into sub-questions"
        ),
    )

    st.markdown("---")

    # ── Retrieval Parameters ──
    st.subheader("🔧 Parameters")
    similarity_top_k = st.slider(
        "Similarity top-K (retrieval)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of candidate chunks retrieved from the vector store.",
    )

    reranker_top_n = st.slider(
        "Re-ranker top-N (post-retrieval)",
        min_value=1,
        max_value=similarity_top_k,
        value=min(3, similarity_top_k),
        help="Number of chunks kept after re-ranking. Only applies to Re-ranking mode.",
        disabled=(technique != "Re-ranking"),
    )

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Place documents in the `data/` folder.")
    st.markdown("2. Enter your Gemini API key above.")
    st.markdown("3. Pick a technique and ask questions!")

# ─── Guard: API key required ─────────────────────────────────────────────────
if not os.environ.get("GEMINI_API_KEY"):
    st.warning("👈 Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()


# ─── Build Index (cached, shared across techniques) ──────────────────────────
@st.cache_resource(show_spinner=False)
def build_index():
    """Load documents from data/ and build a VectorStoreIndex.

    Returns the index object (or None if no documents found).
    """
    with st.spinner("📚 Loading documents and building index …"):
        # Configure global settings
        Settings.llm = GoogleGenAI(model="gemini-2.5-flash-preview-04-17")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        if not os.path.exists("data"):
            os.makedirs("data")

        try:
            documents = SimpleDirectoryReader("data").load_data()
            if not documents:
                return None
        except ValueError:
            return None

        return VectorStoreIndex.from_documents(documents=documents)


index = build_index()

if index is None:
    st.error("No documents found in the `data/` folder. Please add files and restart.")
    st.stop()


# ─── Build query engine for the selected technique ───────────────────────────
def get_query_engine(technique: str, top_k: int, top_n: int):
    """Construct a LlamaIndex query engine based on the selected technique."""

    # Baseline engine — always needed as the foundation
    base_engine = index.as_query_engine(similarity_top_k=top_k)

    if technique == "Baseline":
        return base_engine

    elif technique == "HyDE":
        # Pre-retrieval: generate a hypothetical document from the query,
        # then embed that hypothetical document for better semantic matching.
        hyde_transform = HyDEQueryTransform(include_original=True)
        return TransformQueryEngine(base_engine, hyde_transform)

    elif technique == "Re-ranking":
        # Post-retrieval: retrieve top-K candidates, then re-rank with a
        # cross-encoder model and keep the top-N most relevant.
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=top_n,
        )
        return index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[reranker],
        )

    elif technique == "Sub-Question":
        # Pre-retrieval: decompose complex multi-part questions into
        # independent sub-questions, answer each, then synthesize.
        query_engine_tools = [
            QueryEngineTool(
                query_engine=base_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description=(
                        "Provides information from the loaded documents. "
                        "Use this tool to answer questions about the document content."
                    ),
                ),
            ),
        ]
        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=True,
        )

    return base_engine


# ─── Technique info badge ─────────────────────────────────────────────────────
TECHNIQUE_INFO = {
    "Baseline": {
        "icon": "📦",
        "description": "Simple vector similarity retrieval — no enhancements.",
        "stage": "—",
    },
    "HyDE": {
        "icon": "🔮",
        "description": (
            "**Hypothetical Document Embeddings** — the LLM first generates a "
            "hypothetical answer to your question, then embeds *that* text for "
            "retrieval. This often yields better semantic matches than embedding "
            "the raw query."
        ),
        "stage": "Pre-retrieval",
    },
    "Re-ranking": {
        "icon": "🏆",
        "description": (
            "**Cross-Encoder Re-ranking** — retrieves top-K candidates with vector "
            "search, then re-scores each (query, chunk) pair using a cross-encoder "
            "model (`BAAI/bge-reranker-v2-m3`) and keeps the top-N most relevant."
        ),
        "stage": "Post-retrieval",
    },
    "Sub-Question": {
        "icon": "🧩",
        "description": (
            "**Sub-Question Decomposition** — complex questions are broken into "
            "simpler sub-questions. Each sub-question is answered independently, "
            "then results are synthesized into a single coherent response."
        ),
        "stage": "Pre-retrieval",
    },
}

info = TECHNIQUE_INFO[technique]
with st.expander(f"{info['icon']} Active Technique: **{technique}** ({info['stage']})", expanded=False):
    st.markdown(info["description"])

# ─── Chat Interface ──────────────────────────────────────────────────────────
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I've loaded your documents. Pick a technique in the sidebar and ask me anything!",
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents …"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build engine for current technique
    with st.chat_message("assistant"):
        with st.spinner(f"Running **{technique}** pipeline …"):
            try:
                engine = get_query_engine(technique, similarity_top_k, reranker_top_n)
                response = engine.query(prompt)

                # Main response
                st.markdown(response.response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.response}
                )

                # Source nodes
                if hasattr(response, "source_nodes") and response.source_nodes:
                    with st.expander(f"📄 Source Nodes ({len(response.source_nodes)} retrieved)", expanded=False):
                        for i, node in enumerate(response.source_nodes, 1):
                            score = getattr(node, "score", None)
                            score_str = f" — score: `{score:.4f}`" if score is not None else ""
                            st.markdown(f"**Chunk {i}**{score_str}")
                            st.markdown(f"> {node.text[:500]}{'…' if len(node.text) > 500 else ''}")
                            st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
