import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Assistant Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .assistant-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        margin: 1.5rem 0;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

# Title of the app
st.markdown('<h1 class="main-header">🤖 Enhanced AI Assistant Chatbot</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🔑 API Configuration")
    
    api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your API key from https://platform.openai.com/account/api-keys"
    )
    
    if st.button("Validate API Key"):
        if api_key:
            try:
                # Test the API key
                test_llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
                test_response = test_llm.invoke("Hello")
                st.session_state.api_key_valid = True
                st.success("✅ API Key is valid!")
            except Exception as e:
                st.session_state.api_key_valid = False
                st.error(f"❌ API Key validation failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter an API key first")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Configuration
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🧠 Model Settings")
    
    engine = st.selectbox(
        "Select OpenAI model",
        ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Choose the AI model for responses"
    )
    
    assistant_type = st.selectbox(
        "Assistant Type",
        ["General Assistant", "Microsoft Assistant", "Technical Support", "Creative Writer"],
        index=0,
        help="Choose the type of assistant personality"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Response Parameters
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🎛️ Response Parameters")
    
    temperature = st.slider(
        "Temperature (creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower values = more factual, higher values = more creative"
    )
    
    max_tokens = st.slider(
        "Max Response Length",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum number of tokens in the response"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Controls
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("💬 Chat Controls")
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prompt templates based on assistant type
def get_prompt_template(assistant_type: str) -> ChatPromptTemplate:
    prompts = {
        "General Assistant": ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user questions. Be concise but thorough."),
            ("user", "Question: {question}")
        ]),
        "Microsoft Assistant": ChatPromptTemplate.from_messages([
            ("system", "Act as Microsoft's AI assistant, providing clear, accurate information on products (Windows, Azure, Office 365), services, events (Build, Ignite), and initiatives (AI, sustainability). Prioritize security: never request/store sensitive data (GDPR/CCPA compliant). Offer troubleshooting steps for common issues; escalate complex cases to support.microsoft.com. Highlight accessibility tools (Windows Narrator). Use a friendly, professional tone aligned with Microsoft's mission (Empower every person). Cite official sources (e.g., Per Microsoft's documentation…). Decline non-Microsoft queries politely. End with: Visit microsoft.com for details."),
            ("user", "Question: {question}")
        ]),
        "Technical Support": ChatPromptTemplate.from_messages([
            ("system", "You are a technical support specialist. Provide step-by-step solutions to technical problems. Ask clarifying questions when needed and always prioritize user safety and data security."),
            ("user", "Question: {question}")
        ]),
        "Creative Writer": ChatPromptTemplate.from_messages([
            ("system", "You are a creative writing assistant. Help users with creative writing tasks, brainstorming, storytelling, and improving their writing. Be imaginative and inspiring while maintaining quality."),
            ("user", "Question: {question}")
        ])
    }
    return prompts.get(assistant_type, prompts["General Assistant"])

def generate_response(question: str, api_key: str, model: str, temperature: float, max_tokens: int, assistant_type: str) -> Optional[str]:
    """Generate response from OpenAI API"""
    try:
        if not api_key:
            return "Please provide a valid OpenAI API key."
        
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        prompt = get_prompt_template(assistant_type)
        chain = prompt | llm | StrOutputParser()
        
        with st.spinner("🤔 Thinking..."):
            response = chain.invoke({"question": question})
        
        logger.info(f"Generated response for question: {question[:50]}...")
        return response
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = generate_response(
                question=user_input,
                api_key=api_key,
                model=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                assistant_type=assistant_type
            )
            
            if response:
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

with col2:
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Model", engine)
    st.metric("Temperature", f"{temperature:.1f}")
    
    # API Key Status
    if api_key:
        if st.session_state.api_key_valid:
            st.success("🟢 API Key Valid")
        else:
            st.warning("🟡 API Key Not Validated")
    else:
        st.error("🔴 No API Key")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📋 Instructions:
1. **Enter your OpenAI API Key** in the sidebar
2. **Validate your API key** using the validation button
3. **Choose your preferred model** and assistant type
4. **Adjust parameters** like temperature and max tokens as needed
5. **Start chatting** using the input field below

### 💡 Tips for better responses:
- Be specific and clear with your questions
- Use the temperature slider to control creativity vs factualness
- Try different assistant types for specialized help
- Clear chat history to start fresh conversations
""")

# Environment info (for debugging in production)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("🔍 Debug Information")
    st.sidebar.text(f"Python version: {os.sys.version}")
    st.sidebar.text(f"Streamlit version: {st.__version__}")
    st.sidebar.text(f"Environment: {os.environ.get('ENVIRONMENT', 'Development')}")