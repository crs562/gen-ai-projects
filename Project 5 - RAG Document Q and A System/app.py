import streamlit as st
import os
import time
import logging
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import zipfile
import shutil
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGDocumentQA:
    def __init__(self):
        self.setup_environment()
        self.setup_llm()
        self.setup_prompt()
        
    def setup_environment(self):
        """Setup environment variables and API keys"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.openai_api_key:
            st.error("⚠️ OpenAI API Key not found. Please set OPENAI_API_KEY in your environment variables.")
            st.stop()
            
        if not self.groq_api_key:
            st.error("⚠️ Groq API Key not found. Please set GROQ_API_KEY in your environment variables.")
            st.stop()
            
        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        os.environ['GROQ_API_KEY'] = self.groq_api_key
        
    def setup_llm(self):
        """Initialize the language model"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama3-8b-8192",
                temperature=0.1
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.stop()
            
    def setup_prompt(self):
        """Setup the prompt template"""
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant specialized in helping students learn coding languages and technologies.
            As a Solution Manager with extensive experience in Software Engineering, Computer Systems Analysis, 
            Web Development, Information Security, Data Science, Database Administration, Computer Support, 
            Computer Network Architecture, Cybersecurity, UX Design, Mobile Development, Cloud Computing, 
            AI, Business Intelligence, Data Analysis, and SAP.
            
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            If you cannot find the answer in the provided context, say "I cannot find this information in the provided documents."
            
            <context>
            {context}
            </context>
            
            Question: {input}
            
            Answer:
            """
        )
        
    def create_research_papers_directory(self):
        """Create research_papers directory if it doesn't exist"""
        research_dir = "research_papers"
        if not os.path.exists(research_dir):
            os.makedirs(research_dir)
            logger.info(f"Created {research_dir} directory")
        return research_dir
        
    def handle_file_upload(self, uploaded_files):
        """Handle uploaded PDF files"""
        research_dir = self.create_research_papers_directory()
        
        # Clear existing files
        for file in os.listdir(research_dir):
            file_path = os.path.join(research_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Save uploaded files
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                file_path = os.path.join(research_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logger.info(f"Saved file: {uploaded_file.name}")
            else:
                st.warning(f"Skipping {uploaded_file.name} - only PDF files are supported")
                
    def create_vector_embedding(self, progress_bar=None):
        """Create vector embeddings from documents"""
        try:
            research_dir = self.create_research_papers_directory()
            
            # Check if directory has PDF files
            pdf_files = [f for f in os.listdir(research_dir) if f.endswith('.pdf')]
            if not pdf_files:
                st.error("No PDF files found in research_papers directory. Please upload some PDF files first.")
                return False
                
            if progress_bar:
                progress_bar.progress(0.2, "Loading documents...")
                
            # Initialize embeddings
            st.session_state.embeddings = OpenAIEmbeddings()
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader(research_dir)
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error("No documents were loaded. Please check your PDF files.")
                return False
                
            if progress_bar:
                progress_bar.progress(0.4, "Splitting documents...")
                
            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Limit to first 50 documents to avoid memory issues
            docs_to_process = st.session_state.docs[:50] if len(st.session_state.docs) > 50 else st.session_state.docs
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs_to_process)
            
            if progress_bar:
                progress_bar.progress(0.8, "Creating vector database...")
                
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            
            if progress_bar:
                progress_bar.progress(1.0, "Vector database created successfully!")
                
            logger.info(f"Vector database created with {len(st.session_state.final_documents)} document chunks")
            return True
            
        except Exception as e:
            st.error(f"Error creating vector embeddings: {str(e)}")
            logger.error(f"Error creating vector embeddings: {str(e)}")
            return False
            
    def get_response(self, user_query: str) -> Optional[dict]:
        """Get response from the RAG system"""
        try:
            if "vectors" not in st.session_state:
                st.error("Vector database not initialized. Please click 'Process Documents' first.")
                return None
                
            # Create chains
            document_chain = create_stuff_documents_chain(self.llm, self.prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Get response
            start_time = time.process_time()
            response = retrieval_chain.invoke({'input': user_query})
            response_time = time.process_time() - start_time
            
            logger.info(f"Response time: {response_time:.2f} seconds")
            
            return {
                'answer': response['answer'],
                'context': response['context'],
                'response_time': response_time
            }
            
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            logger.error(f"Error getting response: {str(e)}")
            return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the RAG system
    rag_system = RAGDocumentQA()
    
    # Main title
    st.title("📚 RAG Document Q&A System")
    st.markdown("### AI-Powered Document Analysis and Question Answering")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        # Process uploaded files
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Handle file upload
                    rag_system.handle_file_upload(uploaded_files)
                    
                    # Create progress bar
                    progress_bar = st.progress(0, "Starting document processing...")
                    
                    # Create vector embeddings
                    if rag_system.create_vector_embedding(progress_bar):
                        st.success("✅ Vector database created successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to create vector database")
        
        # System status
        st.header("🔍 System Status")
        if "vectors" in st.session_state:
            st.success("✅ Vector Database Ready")
            doc_count = len(st.session_state.final_documents) if "final_documents" in st.session_state else 0
            st.info(f"📄 {doc_count} document chunks processed")
        else:
            st.warning("⚠️ Vector Database Not Ready")
            
        # API Status
        st.header("🔑 API Status")
        openai_key = "✅ Connected" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
        groq_key = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Not Set"
        st.write(f"OpenAI: {openai_key}")
        st.write(f"Groq: {groq_key}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Query input
        user_query = st.text_area(
            "Enter your question about the uploaded documents:",
            height=100,
            placeholder="e.g., What are the main concepts discussed in the documents?"
        )
        
        # Submit button
        if st.button("🔍 Ask Question", type="primary", disabled=not user_query):
            if "vectors" not in st.session_state:
                st.error("Please upload and process documents first.")
            else:
                with st.spinner("Analyzing documents and generating response..."):
                    response = rag_system.get_response(user_query)
                    
                    if response:
                        st.success("✅ Response Generated!")
                        
                        # Display response
                        st.markdown("### 📝 Answer:")
                        st.markdown(response['answer'])
                        
                        # Response time
                        st.caption(f"⏱️ Response time: {response['response_time']:.2f} seconds")
                        
                        # Document similarity search
                        with st.expander("📋 Source Documents", expanded=False):
                            for i, doc in enumerate(response['context']):
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content)
                                st.markdown("---")
    
    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. **Upload PDFs**: Use the sidebar to upload your PDF documents
        2. **Process**: Click "Process Documents" to create the vector database
        3. **Ask Questions**: Enter your questions in the text area
        4. **Get Answers**: Click "Ask Question" to get AI-powered responses
        """)
        
        st.header("🎯 Features")
        st.markdown("""
        - **Multi-document support**
        - **Real-time processing**
        - **Source attribution**
        - **Fast vector search**
        - **Educational focus**
        """)
        
        st.header("🔧 Technical Stack")
        st.markdown("""
        - **Framework**: Streamlit
        - **LLM**: Groq (Llama3-8b-8192)
        - **Embeddings**: OpenAI
        - **Vector Store**: FAISS
        - **Document Processing**: LangChain
        """)

if __name__ == "__main__":
    main()