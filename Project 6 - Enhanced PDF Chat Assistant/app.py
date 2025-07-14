import os
import tempfile
import streamlit as st
import logging
from typing import Optional
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .error-message {
        background-color: #fee;
        border-left-color: #f56565;
        color: #c53030;
    }
    .success-message {
        background-color: #f0fff4;
        border-left-color: #48bb78;
        color: #2f855a;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def validate_api_key(api_key: str) -> bool:
    """Validate Groq API key format"""
    return api_key and api_key.startswith('gsk_') and len(api_key) > 20

def process_uploaded_files(uploaded_files, progress_callback=None):
    """Process uploaded PDF files with error handling"""
    documents = []
    processed_files = []
    failed_files = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            if progress_callback:
                progress_callback(f"Processing {uploaded_file.name}...")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            
            # Load and process PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            if not docs:
                failed_files.append(f"{uploaded_file.name} (empty or corrupted)")
                continue
                
            documents.extend(docs)
            processed_files.append(uploaded_file.name)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info(f"Successfully processed: {uploaded_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            failed_files.append(f"{uploaded_file.name} ({str(e)})")
            continue
    
    return documents, processed_files, failed_files

def create_vectorstore(documents, progress_callback=None):
    """Create vector store from documents"""
    try:
        if progress_callback:
            progress_callback("Creating embeddings...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if progress_callback:
            progress_callback("Splitting documents...")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("No text chunks created from documents")
        
        if progress_callback:
            progress_callback("Building vector database...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=None  # In-memory for Render deployment
        )
        
        logger.info(f"Created vector store with {len(splits)} chunks")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise e

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_rag_chain(llm, vectorstore):
    """Create the RAG chain"""
    try:
        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. 
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            contextualize_q_prompt
        )
        
        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Be concise and helpful in your responses.
            
            Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create document chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Create RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise e

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 PDF Chat Assistant</h1>
        <p>Upload PDF files and chat with their content using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key (starts with 'gsk_')",
            placeholder="gsk_..."
        )
        
        # Session ID
        session_id = st.text_input(
            "Session ID",
            value="default_session",
            help="Change this to start a new conversation"
        )
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            if session_id in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            st.rerun()
        
        # Reset application button
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # Information
        st.info("💡 **How to use:**\n1. Enter your Groq API key\n2. Upload PDF files\n3. Start chatting!")
        
        # Error display
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Documents")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once"
        )
        
        # Process files button
        if uploaded_files and not st.session_state.documents_processed:
            if st.button("🚀 Process Documents", type="primary"):
                with st.status("Processing your documents...", expanded=True) as status:
                    try:
                        # Process uploaded files
                        documents, processed_files, failed_files = process_uploaded_files(
                            uploaded_files,
                            lambda msg: status.write(msg)
                        )
                        
                        if not documents:
                            st.error("No documents could be processed successfully.")
                            st.session_state.error_message = "Failed to process any documents"
                            return
                        
                        # Create vector store
                        vectorstore = create_vectorstore(
                            documents,
                            lambda msg: status.write(msg)
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents_processed = True
                        st.session_state.processing_complete = True
                        
                        status.update(
                            label="✅ Documents processed successfully!",
                            state="complete"
                        )
                        
                        # Show results
                        if processed_files:
                            st.success(f"✅ Successfully processed: {', '.join(processed_files)}")
                        
                        if failed_files:
                            st.warning(f"⚠️ Failed to process: {', '.join(failed_files)}")
                            
                    except Exception as e:
                        logger.error(f"Error in document processing: {str(e)}")
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.error_message = str(e)
        
        # Show processing status
        if st.session_state.documents_processed:
            st.success("✅ Documents ready for chat!")
    
    with col2:
        st.subheader("💬 Chat Interface")
        
        # Validate inputs
        if not api_key:
            st.warning("🔑 Please enter your Groq API key in the sidebar")
            return
        
        if not validate_api_key(api_key):
            st.error("❌ Invalid API key format. Groq API keys start with 'gsk_'")
            return
        
        if not st.session_state.vectorstore:
            st.info("📄 Upload and process PDF files to start chatting")
            return
        
        try:
            # Initialize LLM
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1000
            )
            
            # Create RAG chain
            rag_chain = create_rag_chain(llm, st.session_state.vectorstore)
            
            # Create conversational chain
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            # Display chat history
            session_history = get_session_history(session_id)
            
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                for message in session_history.messages:
                    with st.chat_message(message.type):
                        st.write(message.content)
            
            # Chat input
            if prompt := st.chat_input("Ask about your documents...", key="chat_input"):
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = conversational_rag_chain.invoke(
                                {"input": prompt},
                                config={"configurable": {"session_id": session_id}}
                            )
                            st.write(response['answer'])
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            st.error(f"Sorry, I encountered an error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error initializing chat: {str(e)}")
            st.error(f"Error initializing chat interface: {str(e)}")

if __name__ == "__main__":
    main()