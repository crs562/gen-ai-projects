import os
import streamlit as st
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import validators
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📚 Document Q&A Assistant")
st.caption("🚀 Upload documents or provide URLs to ask questions powered by OpenAI and LangChain")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Model selection
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    selected_model = st.selectbox(
        "Choose Model",
        model_options,
        index=0
    )
    
    # Temperature setting
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Document processing options
    st.header("📄 Document Options")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Functions for document processing
@st.cache_resource
def initialize_llm(model_name, temp, api_key):
    """Initialize the LLM with error handling"""
    try:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key=api_key
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

@st.cache_resource
def initialize_embeddings(api_key):
    """Initialize embeddings with error handling"""
    try:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIEmbeddings(openai_api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded files and return documents"""
    try:
        documents = []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return []
        
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return []

def process_url(url):
    """Process URL and return documents"""
    try:
        if not validators.url(url):
            st.error("Please enter a valid URL")
            return []
        
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return []

def create_vector_store(documents, embeddings, chunk_size, chunk_overlap):
    """Create vector store from documents"""
    try:
        if not documents:
            st.error("No documents to process")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_response(question, vectorstore, llm):
    """Get response from the Q&A chain"""
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context. 
            If you cannot answer the question based on the context, say so clearly.
            
            Context:
            {context}
            
            Question: {input}
            
            Answer:
            """
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Get response
        response = retrieval_chain.invoke({"input": question})
        return response['answer']
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main application layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 Document Input")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files"
    )
    
    # URL input
    url_input = st.text_input(
        "Or enter a URL",
        placeholder="https://example.com/document"
    )
    
    # Process button
    if st.button("🔄 Process Documents", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
        else:
            with st.spinner("Processing documents..."):
                all_documents = []
                
                # Process uploaded files
                if uploaded_files:
                    for file in uploaded_files:
                        docs = process_uploaded_file(file)
                        all_documents.extend(docs)
                
                # Process URL
                if url_input:
                    docs = process_url(url_input)
                    all_documents.extend(docs)
                
                if all_documents:
                    # Initialize embeddings
                    embeddings = initialize_embeddings(api_key)
                    
                    if embeddings:
                        # Create vector store
                        vectorstore = create_vector_store(
                            all_documents, 
                            embeddings, 
                            chunk_size, 
                            chunk_overlap
                        )
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.documents_processed = True
                            st.success(f"✅ Successfully processed {len(all_documents)} documents!")
                        else:
                            st.error("Failed to create vector store")
                    else:
                        st.error("Failed to initialize embeddings")
                else:
                    st.error("Please upload files or enter a URL")

with col2:
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
        elif not st.session_state.documents_processed:
            st.error("Please process documents first")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = initialize_llm(selected_model, temperature, api_key)
                    
                    if llm and st.session_state.vectorstore:
                        response = get_response(prompt, st.session_state.vectorstore, llm)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Error: Unable to generate response. Please check your API key and try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("- Upload multiple documents for comprehensive Q&A")
st.markdown("- Use specific questions for better answers")
st.markdown("- Adjust chunk size and overlap for optimal results")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.documents_processed = False
    st.rerun()