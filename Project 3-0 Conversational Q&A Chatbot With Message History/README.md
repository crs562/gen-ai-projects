# LangChain Multi-Model Chat Application
A comprehensive chat application built with LangChain, Streamlit, and support for both OpenAI and local Ollama models. Features document processing, RAG (Retrieval-Augmented Generation), and LangSmith tracking.
Features

1. Multi-Model Support: Choose between OpenAI models (GPT-4, GPT-3.5) and local Ollama models
2. Document Processing: Upload files, process web URLs, Wikipedia articles, and ArXiv papers
3. RAG Implementation: Chat with your documents using retrieval-augmented generation
4. LangSmith Integration: Optional tracking and monitoring of your LLM interactions
5. Modern UI: Clean, responsive interface with custom styling
6. Session Management: Persistent chat history during your session

## Installation
1. Clone or Download
2. Save the application files to your local directory:

app.py (main application)
requirements.txt (dependencies)
README.md (this file)

2. Install Dependencies
bashpip install -r requirements.txt
3. Set Up Environment Variables (Optional)
Create a .env file in your project directory:
envOPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name
Running the Application
## Local Development
bashstreamlit run app.py
The application will be available at http://localhost:8501