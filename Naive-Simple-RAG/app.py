import os
import streamlit as st
import nest_asyncio

# Apply nest_asyncio to prevent event loop issues between Streamlit and LlamaIndex
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure the Streamlit page
st.set_page_config(page_title="Naive RAG App", page_icon="🦙", layout="centered")
st.title("🦙 Naive RAG Document Chat")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get your key from Google AI Studio"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Ensure your documents are in the `data/` folder.")
    st.markdown("2. Enter your Gemini API Key above.")
    st.markdown("3. Ask questions about your documents!")

# Stop the app if API key is not provided
if not os.environ.get("GEMINI_API_KEY"):
    st.warning("👈 Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()

# --- Initialize Engine ---
# We use @st.cache_resource so LlamaIndex only loads and embeds the documents ONCE 
# when the app starts, rather than every time the user sends a chat message.
@st.cache_resource(show_spinner=False)
def initialize_engine():
    with st.spinner("Initializing models and loading documents..."):
        # Configure LLM and Embeddings exactly as you had in your notebook
        Settings.llm = GoogleGenAI(model="gemini-3-flash-preview")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Check if data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Load documents
        try:
            documents = SimpleDirectoryReader("data").load_data()
            if not documents:
                return None
        except ValueError:
            # SimpleDirectoryReader throws ValueError if directory is empty
            return None
            
        # Create Index and return Query Engine
        index = VectorStoreIndex.from_documents(documents=documents)
        return index.as_query_engine(similarity_top_k=3)

query_engine = initialize_engine()

# Stop if no documents were found
if query_engine is None:
    st.error("No documents found in the `data` folder. Please add some files and restart the app.")
    st.stop()

# --- Chat Interface ---
# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I've loaded your documents. What would you like to know about them?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Query the RAG engine
                response = query_engine.query(prompt)
                
                # Display response
                st.markdown(response.response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
