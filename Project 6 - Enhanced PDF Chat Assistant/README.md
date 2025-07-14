# PDF Chat Assistant 📄

An intelligent PDF document chat application built with Streamlit, LangChain, and Groq LLM. Upload PDF files and chat with their content using natural language.

## ✨ Features

- **Multi-PDF Support**: Upload and process multiple PDF files simultaneously
- **Intelligent Chat**: Ask questions about your documents using natural language
- **Conversation Memory**: Maintains context across multiple questions in a session
- **Modern UI**: Clean, responsive interface with custom styling
- **Error Handling**: Robust error handling and user feedback
- **Session Management**: Multiple conversation sessions with unique IDs
- **Cloud Deployment**: Ready for deployment on Render or other cloud platforms

## 🚀 Technologies Used

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Document Processing**: LangChain + PyPDF
- **Deployment**: Docker + Render

## 📋 Prerequisites

- Python 3.10+
- Groq API Key ([Get one here](https://console.groq.com/))
- Git

## 🔧 Local Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🌐 Deploy to Render

### Step 1: Prepare Your Repository

1. **Fork or clone this repository** to your GitHub account
2. **Push your code** to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Create Render Account

1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub account

### Step 3: Deploy Web Service

1. **Click "New +"** and select **"Web Service"**
2. **Connect your repository** containing the PDF Chat Assistant code
3. **Configure the service**:
   - **Name**: `pdf-chat-assistant`
   - **Runtime**: `Docker`
   - **Branch**: `main`
   - **Build Command**: Leave empty (Docker handles this)
   - **Start Command**: Leave empty (Docker handles this)

### Step 4: Set Environment Variables

In the Render dashboard, add these environment variables:

**Required:**
- `GROQ_API_KEY`: Your Groq API key

**Optional (already set in render.yaml):**
- `STREAMLIT_SERVER_PORT`: `8501`
- `STREAMLIT_SERVER_ADDRESS`: `0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS`: `true`

### Step 5: Deploy

1. **Click "Create Web Service"**
2. **Wait for deployment** (usually 5-10 minutes)
3. **Access your app** via the provided Render URL

### Alternative: One-Click Deploy

Use the render.yaml file for one-click deployment:

1. **Fork this repository**
2. **Connect to Render**
3. **Import from render.yaml**
4. **Set your GROQ_API_KEY** in environment variables
5. **Deploy**

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key (required) | - |
| `GROQ_MODEL_NAME` | Groq model to use | `llama-3.3-70b-versatile` |
| `MODEL_TEMPERATURE` | Model temperature (0-1) | `0.3` |
| `MAX_TOKENS` | Maximum response tokens | `1000` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `RETRIEVAL_K` | Number of chunks to retrieve | `5` |

### Customizing the Application

1. **Styling**: Modify the CSS in the `st.markdown()` sections
2. **Model Settings**: Change the Groq model or temperature
3. **Chunking Strategy**: Adjust chunk size and overlap for your documents
4. **UI Layout**: Modify the Streamlit layout and components

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure your Groq API key starts with `gsk_`
   - Check your API key is valid and has credits

2. **PDF Processing Error**:
   - Ensure PDFs are not password-protected
   - Check PDFs contain extractable text (not just images)

3. **Memory Issues**:
   - Large PDFs may cause memory issues on free tiers
   - Consider upgrading your Render plan or splitting large documents

4. **Deployment Failures**:
   - Check Render logs for specific error messages
   - Ensure all required files are in your repository
   - Verify environment variables are set correctly

### Performance Optimization

1. **Document Size**: Limit PDF size to under 50MB
2. **Chunk Strategy**: Adjust chunk size based on your document types
3. **Model Selection**: Use smaller models for faster responses
4. **Caching**: Implement caching for frequently accessed documents


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for the RAG implementation
- [Groq](https://groq.com/) for fast LLM inference
- [HuggingFace](https://huggingface.co/) for embeddings
- [Render](https://render.com/) for easy deployment