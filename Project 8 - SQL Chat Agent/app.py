import streamlit as st
import os
from pathlib import Path
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SQL Chat Agent", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
    <div class="main-header">
        <h1>🤖 SQL Chat Agent</h1>
        <p>Powered by LangChain, Groq, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Constants
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

def initialize_database():
    """Initialize the SQLite database with sample data if it doesn't exist."""
    try:
        db_path = Path(__file__).parent / "student.db"
        if not db_path.exists():
            logger.info("Creating student database...")
            
            connection = sqlite3.connect(str(db_path))
            cursor = connection.cursor()
            
            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS STUDENT (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    NAME VARCHAR(25) NOT NULL,
                    CLASS VARCHAR(25),
                    SECTION VARCHAR(1),
                    MARKS INT CHECK(MARKS >= 0 AND MARKS <= 100)
                )
            """)
            
            # Sample data
            students = [
                ('Alice', 'Data Science', 'A', 85),
                ('Bob', 'Data Science', 'B', 90),
                ('Charlie', 'AI Agents', 'A', 75),
                ('David', 'AI Agents', 'B', 80),
                ('Eve', 'DevOps', 'A', 95),
                ('Frank', 'DevOps', 'B', 70),
                ('Grace', 'Data Science', 'A', 88),
                ('Heidi', 'Data Science', 'B', 92),
                ('Ivan', 'AI Agents', 'A', 78),
                ('Judy', 'AI Agents', 'B', 82),
                ('Karl', 'DevOps', 'A', 97),
                ('Leo', 'DevOps', 'B', 65),
                ('Mallory', 'Data Science', 'A', 89),
                ('Nina', 'Data Science', 'B', 91),
                ('Oscar', 'AI Agents', 'A', 76),
                ('Peggy', 'AI Agents', 'B', 84),
                ('Quentin', 'DevOps', 'A', 98),
                ('Rupert', 'DevOps', 'B', 68),
                ('Sybil', 'Data Science', 'A', 87),
                ('Trent', 'Data Science', 'B', 93)
            ]
            
            cursor.executemany("INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) VALUES (?,?,?,?)", students)
            connection.commit()
            connection.close()
            
            logger.info("Database created successfully!")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        st.error(f"Error initializing database: {str(e)}")

# Initialize database on startup
initialize_database()

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Database selection
    st.subheader("Database Settings")
    radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to MySQL Database"]
    selected_opt = st.radio(
        label="Choose the database you want to chat with:",
        options=radio_opt,
        help="Select your preferred database connection type"
    )
    
    # Database configuration
    if radio_opt.index(selected_opt) == 1:
        db_uri = MYSQL
        st.info("🔗 MySQL Connection")
        mysql_host = st.text_input("MySQL Host", placeholder="localhost")
        mysql_user = st.text_input("MySQL User", placeholder="root")
        mysql_password = st.text_input("MySQL Password", type="password")
        mysql_db = st.text_input("MySQL Database", placeholder="mydb")
    else:
        db_uri = LOCALDB
        st.info("📁 Using local SQLite database")
    
    st.divider()
    
    # API Key configuration
    st.subheader("API Settings")
    api_key = st.text_input(
        label="Groq API Key",
        type="password",
        help="Enter your Groq API key. Get one from https://console.groq.com/",
        placeholder="gsk_..."
    )
    
    # Use environment variable if API key not provided
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ Using API key from environment")
    
    # Model selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        help="Choose the LLM model for SQL query generation"
    )
    
    st.divider()
    
    # Action buttons
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you query the database?"}]
        st.rerun()
    
    # Database info
    with st.expander("📊 Database Schema"):
        st.code("""
        STUDENT Table:
        - ID: INTEGER (Primary Key)
        - NAME: VARCHAR(25)
        - CLASS: VARCHAR(25) 
        - SECTION: VARCHAR(1)
        - MARKS: INT (0-100)
        
        Sample Classes:
        - Data Science
        - AI Agents  
        - DevOps
        """)

# Validation
error_messages = []

if not api_key:
    error_messages.append("Please provide a Groq API key in the sidebar or set GROQ_API_KEY environment variable")

if db_uri == MYSQL and not all([mysql_host, mysql_user, mysql_password, mysql_db]):
    error_messages.append("Please provide all MySQL connection details")

if error_messages:
    for msg in error_messages:
        st.error(msg)
    st.stop()

# Database configuration function
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    """Configure database connection with error handling."""
    try:
        if db_uri == LOCALDB:
            dbfilepath = (Path(__file__).parent / "student.db").absolute()
            if not dbfilepath.exists():
                raise FileNotFoundError(f"Database file not found: {dbfilepath}")
            
            creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        
        elif db_uri == MYSQL:
            if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
                raise ValueError("Missing MySQL connection parameters")
            
            connection_string = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            return SQLDatabase(create_engine(connection_string))
    
    except Exception as e:
        logger.error(f"Database configuration error: {str(e)}")
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

# Initialize LLM and database
try:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=selected_model,
        streaming=True,
        temperature=0
    )
    
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)
    
    # Create toolkit and agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
    
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    logger.error(f"Initialization error: {traceback.format_exc()}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your SQL Chat Agent. I can help you query the student database. Try asking questions like:\n- 'Show me all students in Data Science class'\n- 'What's the average score by class?'\n- 'Who are the top 5 students with highest marks?'"}
    ]

# Chat interface
st.subheader("💬 Chat with your Database")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_query = st.chat_input(
    placeholder="Ask anything about the database... e.g., 'Show me students with marks above 90'"
)

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing your query and generating SQL..."):
                streamlit_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(user_query, callbacks=[streamlit_callback])
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            logger.error(f"Query error: {traceback.format_exc()}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🔧 Built with Streamlit")
with col2:
    st.info("🤖 Powered by Groq")
with col3:
    st.info("🔗 Using LangChain")

# Display connection status
if db_uri == LOCALDB:
    st.success("✅ Connected to SQLite database")
else:
    st.success(f"✅ Connected to MySQL: {mysql_host}/{mysql_db}")