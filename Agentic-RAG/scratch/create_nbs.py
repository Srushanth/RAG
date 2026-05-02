import nbformat as nbf
import os

# Create notebooks directory
os.makedirs('notebooks', exist_ok=True)

def create_notebook(filename, title, cells):
    nb = nbf.v4.new_notebook()
    nb_cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    
    for cell_type, content in cells:
        if cell_type == 'markdown':
            nb_cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            nb_cells.append(nbf.v4.new_code_cell(content))
            
    nb['cells'] = nb_cells
    
    with open(f'notebooks/{filename}', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

# ---------------------------------------------------------
# 1. LlamaIndex Agents
# ---------------------------------------------------------
create_notebook(
    "01_llamaindex_agents.ipynb",
    "🦙 Agentic RAG with LlamaIndex",
    [
        ('markdown', "LlamaIndex makes it incredibly easy to turn query engines into tools for an agent. Here we use the `ReActAgent`."),
        ('code', "import os\nimport nest_asyncio\n\nnest_asyncio.apply()\nos.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')"),
        ('code', "from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex\nfrom llama_index.llms.google_genai import GoogleGenAI\nfrom llama_index.embeddings.huggingface import HuggingFaceEmbedding\n\nSettings.llm = GoogleGenAI(model='gemini-2.5-flash')\nSettings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')"),
        ('code', "print('Loading Data...')\ndocuments = SimpleDirectoryReader('../data').load_data()\nprint('Building Index...')\nindex = VectorStoreIndex.from_documents(documents)\nquery_engine = index.as_query_engine(similarity_top_k=3)"),
        ('code', "from llama_index.core.tools import QueryEngineTool, ToolMetadata\n\nvector_tool = QueryEngineTool(\n    query_engine=query_engine,\n    metadata=ToolMetadata(\n        name='apple_env_report',\n        description='Useful for answering questions about Apple\\'s environmental goals and progress.'\n    )\n)"),
        ('code', "from llama_index.core.agent import ReActAgent\n\nagent = ReActAgent.from_tools([vector_tool], llm=Settings.llm, verbose=True)\n\nresponse = agent.chat('What are Apple\\'s goals for 2030 and how are they achieving them?')\nprint(response)")
    ]
)

# ---------------------------------------------------------
# 2. LangGraph
# ---------------------------------------------------------
create_notebook(
    "02_langgraph.ipynb",
    "🦜🕸️ Agentic RAG with LangGraph",
    [
        ('markdown', "LangGraph allows us to define agents as state graphs. This gives us precise control over the flow of execution."),
        ('code', "import os\nos.environ['GOOGLE_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')"),
        ('code', "from langchain_google_genai import ChatGoogleGenerativeAI\nfrom langchain_core.tools import tool\nfrom pypdf import PdfReader\n\n# Set up Gemini\nllm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')"),
        ('code', "@tool\ndef search_report(query: str) -> str:\n    \"\"\"Search the Apple Environmental Report for information.\"\"\"\n    # A naive exact match search for demonstration\n    reader = PdfReader('../data/Apple_Environmental_Progress_Report_2025.pdf')\n    text = ''\n    for page in reader.pages[:10]: # Read first 10 pages for speed\n        text += page.extract_text() + '\\n'\n    \n    # In a real app, use a Vector Store here.\n    return text[:2000] # Return a snippet\n\ntools = [search_report]\nllm_with_tools = llm.bind_tools(tools)"),
        ('code', "from typing import Annotated\nfrom typing_extensions import TypedDict\nfrom langgraph.graph import StateGraph, START, END\nfrom langgraph.graph.message import add_messages\nfrom langgraph.prebuilt import ToolNode, tools_condition\n\nclass State(TypedDict):\n    messages: Annotated[list, add_messages]\n\ndef chatbot(state: State):\n    return {'messages': [llm_with_tools.invoke(state['messages'])]}\n\ngraph_builder = StateGraph(State)\ngraph_builder.add_node('chatbot', chatbot)\n\ntool_node = ToolNode(tools=[search_report])\ngraph_builder.add_node('tools', tool_node)\n\ngraph_builder.add_conditional_edges('chatbot', tools_condition)\ngraph_builder.add_edge('tools', 'chatbot')\ngraph_builder.add_edge(START, 'chatbot')\n\ngraph = graph_builder.compile()"),
        ('code', "from langchain_core.messages import HumanMessage\n\nfor event in graph.stream({'messages': [HumanMessage(content='What are Apple\\'s environmental goals?')] }):\n    for value in event.values():\n        print('Agent:', value['messages'][-1].content)")
    ]
)

# ---------------------------------------------------------
# 3. CrewAI
# ---------------------------------------------------------
create_notebook(
    "03_crewai.ipynb",
    "🚣 Agentic RAG with CrewAI",
    [
        ('markdown', "CrewAI focuses on multi-agent collaboration, allowing you to define 'roles' and 'goals' for each agent."),
        ('code', "import os\nos.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')\nos.environ['GOOGLE_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')"),
        ('code', "from crewai import Agent, Task, Crew, Process, LLM\nfrom crewai_tools import PDFSearchTool\n\n# Configure Gemini LLM\nllm = LLM(model='gemini/gemini-2.5-flash')\n\n# Initialize the PDFSearchTool\npdf_tool = PDFSearchTool(pdf='../data/Apple_Environmental_Progress_Report_2025.pdf')"),
        ('code', "researcher = Agent(\n    role='Environmental Report Researcher',\n    goal='Extract accurate data regarding environmental initiatives from the provided report.',\n    backstory='You are an expert analyst who reads corporate environmental reports to extract key metrics and goals.',\n    tools=[pdf_tool],\n    llm=llm,\n    verbose=True\n)\n\nsummarizer = Agent(\n    role='Executive Summarizer',\n    goal='Condense the researcher\\'s findings into a concise, actionable summary.',\n    backstory='You are a skilled executive assistant who summarizes complex technical data into easy-to-read reports.',\n    llm=llm,\n    verbose=True\n)"),
        ('code', "task1 = Task(\n    description='Search the Apple Environmental Report to find all goals related to 2030.',\n    expected_output='A raw list of goals and metrics for 2030 found in the report.',\n    agent=researcher\n)\n\ntask2 = Task(\n    description='Take the raw list of goals and write a 3-bullet point executive summary.',\n    expected_output='A 3-bullet point markdown summary of Apple\\'s 2030 goals.',\n    agent=summarizer\n)"),
        ('code', "crew = Crew(\n    agents=[researcher, summarizer],\n    tasks=[task1, task2],\n    process=Process.sequential,\n    verbose=True\n)\n\nresult = crew.kickoff()\nprint('######################\\nFINAL RESULT\\n######################')\nprint(result)")
    ]
)

# ---------------------------------------------------------
# 4. Google ADK / GenAI
# ---------------------------------------------------------
create_notebook(
    "04_google_adk.ipynb",
    "🛠️ Agentic RAG with Google GenAI SDK",
    [
        ('markdown', "We can use the official `google-genai` SDK to build an agent that has access to tools (functions)."),
        ('code', "import os\nimport nest_asyncio\n\nnest_asyncio.apply()\nos.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')"),
        ('code', "from google import genai\nfrom google.genai import types\nfrom pypdf import PdfReader\n\nclient = genai.Client()\n\n# Define our tool as a standard Python function\ndef read_report_snippet(page_num: int) -> str:\n    \"\"\"Reads a specific page from the Apple Environmental Report.\"\"\"\n    try:\n        reader = PdfReader('../data/Apple_Environmental_Progress_Report_2025.pdf')\n        if page_num < len(reader.pages):\n            return reader.pages[page_num].extract_text()\n        return 'Page not found.'\n    except Exception as e:\n        return str(e)"),
        ('code', "print('Initializing chat with tools...')\nchat = client.chats.create(\n    model='gemini-2.5-flash',\n    config=types.GenerateContentConfig(\n        tools=[read_report_snippet],\n        temperature=0.0\n    )\n)\n\n# The agent will decide to call `read_report_snippet`\nresponse = chat.send_message('Can you read page 5 of the environmental report and summarize its contents?')\nprint(response.text)")
    ]
)

# ---------------------------------------------------------
# 5. smolagents
# ---------------------------------------------------------
create_notebook(
    "05_strands_agents.ipynb",
    "🤗 Agentic RAG with smolagents",
    [
        ('markdown', "Smolagents (by Hugging Face) is a highly minimal, code-centric agent framework. We use the `CodeAgent` which writes python code to solve tasks."),
        ('code', "import os\nos.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')"),
        ('code', "from smolagents import CodeAgent, LiteLLMModel, tool\nfrom pypdf import PdfReader\n\n# Configure the Gemini model via LiteLLM\nmodel = LiteLLMModel('gemini/gemini-2.5-flash')\n\n@tool\ndef extract_pdf_text(filepath: str, start_page: int, num_pages: int) -> str:\n    \"\"\"\n    Extracts text from a PDF file.\n    Args:\n        filepath: The path to the PDF file (e.g. '../data/Apple_Environmental_Progress_Report_2025.pdf')\n        start_page: The starting page index (0-based)\n        num_pages: The number of pages to read\n    \"\"\"\n    reader = PdfReader(filepath)\n    text = ''\n    end_page = min(start_page + num_pages, len(reader.pages))\n    for i in range(start_page, end_page):\n        text += reader.pages[i].extract_text() + '\\n'\n    return text"),
        ('code', "agent = CodeAgent(tools=[extract_pdf_text], model=model, add_base_tools=True)\n\n# Ask the agent a question that requires reading the report\nagent.run('Use the extract_pdf_text tool to read the first 5 pages of ../data/Apple_Environmental_Progress_Report_2025.pdf and tell me what the main topic is.')")
    ]
)
