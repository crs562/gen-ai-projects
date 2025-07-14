import streamlit as st
import pandas as pd
import os
import time
import logging
from typing import Optional, Tuple, List
from datetime import datetime

# Import LangChain components
try:
    from langchain_groq import ChatGroq
    from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
    from langchain.agents import initialize_agent, AgentType
    from langchain.callbacks import StreamlitCallbackHandler
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from PyPDF2 import PdfReader
except ImportError as e:
    st.error(f"Required package not installed: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="🔍 Smart Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SmartResearchAssistant:
    def __init__(self):
        self.initialize_session_state()
        self.llm = None
        self.agent = None
        self.vector_db = None
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'messages': [{"role": "assistant", "content": "Hello! I'm your Smart Research Assistant. I can search academic papers, Wikipedia, the web, and analyze your uploaded documents. How can I help you today?"}],
            'processed_data': None,
            'vector_db': None,
            'agent_initialized': False,
            'uploaded_files': [],
            'search_history': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def safe_search(self, search_tool, query: str, max_retries: int = 2) -> Tuple[bool, str]:
        """Enhanced search with retries and error handling"""
        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Rate limiting
                result = search_tool.run(query)
                return True, result
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return False, f"Search failed after {max_retries} attempts: {str(e)}"
                time.sleep(2 * (attempt + 1))  # Exponential backoff
        return False, "Unknown search error"

    def initialize_agent(self, api_key: str) -> Tuple[bool, str]:
        """Initialize the LangChain agent with proper error handling"""
        try:
            # Initialize LLM
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-8b-8192",
                temperature=0.3,
                max_tokens=2048
            )
            
            # Configure search tools
            arxiv_wrapper = ArxivAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=800,
                load_max_docs=2
            )
            arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
            
            wiki_wrapper = WikipediaAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=800
            )
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
            
            web_search_tool = DuckDuckGoSearchRun(name="Web_Search")
            
            tools = [arxiv_tool, wiki_tool, web_search_tool]
            
            # Initialize agent
            self.agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                max_iterations=5,
                verbose=True
            )
            
            st.session_state.agent_initialized = True
            return True, "Research agent initialized successfully!"
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            return False, f"Failed to initialize agent: {str(e)}"

    def process_pdf(self, uploaded_file) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            text = ""
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else None
        except Exception as e:
            st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            return None

    def process_csv(self, uploaded_file) -> Optional[str]:
        """Convert CSV to text format for analysis"""
        try:
            df = pd.read_csv(uploaded_file)
            # Create a summary of the CSV
            summary = f"CSV File: {uploaded_file.name}\n"
            summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            summary += "Sample data:\n"
            summary += df.head(10).to_string(index=False)
            return summary
        except Exception as e:
            st.error(f"Error processing CSV {uploaded_file.name}: {str(e)}")
            return None

    def create_vector_database(self, text: str) -> Optional[FAISS]:
        """Create vector database for document search"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            if not chunks:
                return None
                
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            vector_db = FAISS.from_texts(chunks, embeddings)
            return vector_db
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return None

    def search_documents(self, query: str, k: int = 3) -> str:
        """Search uploaded documents using vector similarity"""
        if not st.session_state.vector_db:
            return "No documents have been uploaded and processed."
        
        try:
            docs = st.session_state.vector_db.similarity_search(query, k=k)
            if not docs:
                return "No relevant information found in your documents."
            
            relevant_content = "\n\n".join([doc.page_content for doc in docs])
            return f"Based on your uploaded documents:\n\n{relevant_content}"
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def render_sidebar(self):
        """Render the sidebar with configuration and file upload"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Key input
            api_key = st.text_input(
                "Groq API Key:",
                type="password",
                help="Get your free API key from https://console.groq.com/keys"
            )
            
            if api_key and not st.session_state.agent_initialized:
                with st.spinner("Initializing research agent..."):
                    success, message = self.initialize_agent(api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            st.divider()
            
            # File upload section
            st.header("📁 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF or CSV files",
                type=['pdf', 'csv'],
                accept_multiple_files=True,
                help="Upload research papers, reports, or data files to analyze"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Documents", type="primary"):
                    self.process_uploaded_files(uploaded_files)
            
            # Display processed files
            if st.session_state.uploaded_files:
                st.subheader("📋 Processed Files")
                for file_info in st.session_state.uploaded_files:
                    st.write(f"✅ {file_info['name']} ({file_info['type']})")
            
            st.divider()
            
            # Search history
            if st.session_state.search_history:
                st.subheader("🕒 Recent Searches")
                for search in st.session_state.search_history[-5:]:
                    st.caption(f"🔍 {search[:50]}...")

    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files"""
        processed_text = ""
        file_info = []
        
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = self.process_pdf(uploaded_file)
                    file_type = "PDF"
                elif uploaded_file.type == "text/csv":
                    text = self.process_csv(uploaded_file)
                    file_type = "CSV"
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.type}")
                    continue
                
                if text:
                    processed_text += f"\n\n--- {uploaded_file.name} ---\n{text}"
                    file_info.append({
                        'name': uploaded_file.name,
                        'type': file_type,
                        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
        
        if processed_text:
            st.session_state.processed_data = processed_text
            st.session_state.uploaded_files = file_info
            
            # Create vector database
            with st.spinner("Creating search index..."):
                st.session_state.vector_db = self.create_vector_database(processed_text)
            
            if st.session_state.vector_db:
                st.success(f"✅ Successfully processed {len(file_info)} files!")
            else:
                st.warning("Files processed but search index creation failed.")
        else:
            st.error("No valid content could be extracted from the uploaded files.")

    def handle_user_query(self, query: str):
        """Process user query and generate response"""
        # Add to search history
        st.session_state.search_history.append(query)
        
        # Check if we should search documents first
        if st.session_state.vector_db and any(word in query.lower() for word in ['document', 'file', 'uploaded', 'my data']):
            response = self.search_documents(query)
        elif st.session_state.agent_initialized and self.agent:
            # Use the research agent
            try:
                with st.spinner("Researching your question..."):
                    st_callback = StreamlitCallbackHandler(
                        st.container(),
                        expand_new_thoughts=False
                    )
                    response = self.agent.run(query, callbacks=[st_callback])
            except Exception as e:
                response = f"I encountered an error while researching: {str(e)}\n\nPlease try rephrasing your question or check your API key."
        else:
            response = "Please configure your Groq API key in the sidebar to enable web and academic search capabilities."
        
        return response

    def render_main_interface(self):
        """Render the main chat interface"""
        # Header
        st.markdown('<h1 class="main-header">🔍 Smart Research Assistant</h1>', unsafe_allow_html=True)
        
        # Feature overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>🔬 Academic Search</h4>
                <p>Search arXiv papers and academic resources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>🌐 Web Research</h4>
                <p>Get latest information from the web and Wikipedia</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>📄 Document Analysis</h4>
                <p>Upload and analyze your PDF and CSV files</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Example queries
        st.subheader("💡 Try these example queries:")
        example_queries = [
            "Latest research on large language models",
            "Explain quantum computing for beginners",
            "Recent developments in renewable energy",
            "What are the key findings in my uploaded documents?"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"📝 {query}", key=f"example_{i}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

    def render_chat_interface(self):
        """Render the chat messages and input"""
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about research, upload documents, or search the web..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                response = self.handle_user_query(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_interface()
        self.render_chat_interface()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>Built with Streamlit, LangChain, and Groq AI • 
            <a href="https://github.com" target="_blank">View Source</a></p>
        </div>
        """, unsafe_allow_html=True)

# Application entry point
def main():
    try:
        app = SmartResearchAssistant()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()