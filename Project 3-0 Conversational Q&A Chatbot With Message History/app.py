import os
import streamlit as st
import tempfile
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    WebBaseLoader, TextLoader, PyPDFLoader, 
    WikipediaLoader, ArxivLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import bs4

# Page configuration
st.set_page_config(
    page_title="LangChain Multi-Model Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.sidebar-content {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196F3;
}
.assistant-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

initialize_session_state()

# Main title
st.markdown('<div class="main-header"><h1>🤖 LangChain Multi-Model Chat Application</h1></div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("⚙️ Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Choose Model Type",
        ["OpenAI", "Ollama (Local)"],
        index=0
    )
    
    if model_type == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        selected_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
    else:
        selected_model = st.selectbox(
            "Ollama Model",
            ["gemma:7b", "llama2", "mistral", "codegemma:latest"],
            index=0
        )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # LangSmith configuration
    st.subheader("📊 LangSmith Tracking")
    enable_langsmith = st.checkbox("Enable LangSmith", value=False)
    if enable_langsmith:
        langsmith_api_key = st.text_input("LangSmith API Key", type="password")
        project_name = st.text_input("Project Name", value="langchain-chat")
        if langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = project_name
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document Processing Section
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("📄 Document Processing")
    
    doc_source = st.selectbox(
        "Document Source",
        ["Upload File", "Web URL", "Wikipedia", "ArXiv"],
        index=0
    )
    
    if doc_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['txt', 'pdf'],
            help="Upload a text or PDF file"
        )
    elif doc_source == "Web URL":
        web_url = st.text_input("Enter Web URL")
    elif doc_source == "Wikipedia":
        wiki_query = st.text_input("Wikipedia Search Query")
        max_docs = st.slider("Max Documents", 1, 5, 2)
    elif doc_source == "ArXiv":
        arxiv_query = st.text_input("ArXiv Paper ID or Query")
        max_docs = st.slider("Max Documents", 1, 5, 2)
    
    if st.button("Process Documents"):
        process_documents(doc_source, locals())
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize LLM
@st.cache_resource
def load_llm(model_type, model_name, temperature):
    try:
        if model_type == "OpenAI":
            return ChatOpenAI(model=model_name, temperature=temperature)
        else:
            return Ollama(model=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Document processing functions
def process_documents(source, variables):
    """Process documents based on the selected source"""
    try:
        documents = []
        
        if source == "Upload File" and 'uploaded_file' in variables and variables['uploaded_file']:
            documents = process_uploaded_file(variables['uploaded_file'])
        elif source == "Web URL" and 'web_url' in variables and variables['web_url']:
            documents = process_web_url(variables['web_url'])
        elif source == "Wikipedia" and 'wiki_query' in variables and variables['wiki_query']:
            documents = process_wikipedia(variables['wiki_query'], variables['max_docs'])
        elif source == "ArXiv" and 'arxiv_query' in variables and variables['arxiv_query']:
            documents = process_arxiv(variables['arxiv_query'], variables['max_docs'])
        
        if documents:
            create_vector_store(documents)
            st.success(f"Successfully processed {len(documents)} documents!")
            st.session_state.documents_loaded = True
        else:
            st.error("No documents were processed. Please check your input.")
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process uploaded file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            st.error("Unsupported file type")
            return []
        
        documents = loader.load()
        return documents
    finally:
        os.unlink(tmp_file_path)

def process_web_url(url):
    """Process web URL"""
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            class_=("post-title", "post-content", "post-header", "content", "main", "article")
        ))
    )
    return loader.load()

def process_wikipedia(query, max_docs):
    """Process Wikipedia search"""
    loader = WikipediaLoader(query=query, load_max_docs=max_docs)
    return loader.load()

def process_arxiv(query, max_docs):
    """Process ArXiv papers"""
    loader = ArxivLoader(query=query, load_max_docs=max_docs)
    return loader.load()

def create_vector_store(documents):
    """Create vector store from documents"""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        if os.getenv("OPENAI_API_KEY"):
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.vector_store = vector_store
        else:
            st.error("OpenAI API key required for document processing")
            
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

# Chat interface
def display_chat_messages():
    """Display chat messages with custom styling"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_response(question, llm):
    """Get response from LLM with optional RAG"""
    try:
        if st.session_state.vector_store is not None:
            # Use RAG if documents are loaded
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based on the provided context. If the context doesn't contain 
            relevant information, answer based on your general knowledge but mention that you're not 
            using the uploaded documents.
            
            Context: {context}
            Question: {input}
            
            Answer:
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({"input": question})
            return response["answer"]
        else:
            # Use simple chain without RAG
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Respond concisely and accurately."),
                ("user", "Question: {question}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"question": question})
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main chat interface
st.subheader("💬 Chat Interface")

# Document status indicator
if st.session_state.documents_loaded:
    st.success("📄 Documents loaded - RAG mode active")
else:
    st.info("💭 Standard chat mode - no documents loaded")

# Display chat messages
display_chat_messages()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get LLM response
    llm = load_llm(model_type, selected_model, temperature)
    
    if llm:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt, llm)
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Failed to load the selected model. Please check your configuration.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚀 Built with LangChain, Streamlit, and ❤️
</div>
""", unsafe_allow_html=True)

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}]
    st.rerun()