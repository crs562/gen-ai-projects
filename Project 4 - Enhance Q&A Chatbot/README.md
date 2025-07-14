# 🤖 Enhanced AI Chatbot - Deployment Guide

A feature-rich AI chatbot application built with Streamlit and OpenAI's API, ready for deployment on Render.

## 📋 Features

- **Multiple AI Models**: Support for GPT-4o, GPT-4-turbo, GPT-4, and GPT-3.5-turbo
- **Assistant Types**: General Assistant, Microsoft Assistant, Technical Support, and Creative Writer
- **Advanced Controls**: Temperature and max tokens adjustment
- **Chat History**: Persistent conversation history during session
- **API Key Validation**: Built-in API key testing
- **Responsive Design**: Modern UI with custom styling
- **Error Handling**: Comprehensive error handling and logging
- **Debug Mode**: Development debugging information

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git (for deployment)

### Local Development

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```
4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
5. **Open your browser** to `http://localhost:8501`

## 🌐 Deployment on Render

### Method 1: Using Render Dashboard (Recommended)

1. **Prepare your repository:**
   - Create a new GitHub repository
   - Upload all the files (app.py, requirements.txt, render.yaml, .streamlit/config.toml)

2. **Deploy on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file

3. **Configure Environment Variables:**
   - In Render dashboard, go to your service
   - Navigate to "Environment" tab
   - Add environment variable:
     - Key: `OPENAI_API_KEY`
     - Value: Your OpenAI API key

4. **Deploy:**
   - Click "Deploy Latest Commit"
   - Wait for deployment to complete
   - Your app will be available at `https://your-app-name.onrender.com`

### Method 2: Using Render CLI

1. **Install Render CLI:**
   ```bash
   npm install -g @render/cli
   ```

2. **Login to Render:**
   ```bash
   render login
   ```

3. **Deploy:**
   ```bash
   render deploy
   ```

### Method 3: Manual Configuration

If you prefer manual configuration:

1. **Create Web Service:**
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

2. **Environment Variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: production
   - `STREAMLIT_SERVER_PORT`: 8501
   - `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0

## 🐳 Docker Deployment (Alternative)

If you prefer using Docker:

1. **Build the image:**
   ```bash
   docker build -t ai-chatbot .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here ai-chatbot
   ```

## 📁 Project Structure

```
ai-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── Dockerfile            # Docker configuration (optional)
├── .env.example          # Environment variables template
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `ENVIRONMENT` | Environment (development/production) | No |
| `LANGCHAIN_API_KEY` | LangChain API key for tracing | No |
| `LANGCHAIN_TRACING_V2` | Enable LangChain tracing | No |
| `LANGCHAIN_PROJECT` | LangChain project name | No |

### Streamlit Configuration

The app includes a `.streamlit/config.toml` file with optimized settings for production deployment.

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Issues:**
   - Ensure your OpenAI API key is valid
   - Check if you have sufficient API credits
   - Use the "Validate API Key" button in the sidebar

2. **Deployment Issues:**
   - Verify all files are in the repository
   - Check environment variables are set correctly
   - Review build logs in Render dashboard

3. **Performance Issues:**
   - Adjust temperature and max tokens for faster responses
   - Use GPT-3.5-turbo for quicker responses
   - Consider upgrading to a paid Render plan for better performance

### Debug Mode

Enable debug mode by checking "Show Debug Info" in the sidebar to see:
- Python version
- Streamlit version
- Environment details

## 🔐 Security Notes

- Never commit your `.env` file with real API keys
- Use environment variables for sensitive data
- The app includes GDPR/CCPA compliance features
- API keys are masked in the UI

## 🎯 Usage Tips

1. **Getting Started:**
   - Enter your OpenAI API key in the sidebar
   - Validate your key using the validation button
   - Choose your preferred assistant type

2. **Optimizing Responses:**
   - Use lower temperature (0.1-0.4) for factual responses
   - Use higher temperature (0.7-1.0) for creative responses
   - Adjust max tokens based on desired response length

3. **Assistant Types:**
   - **General Assistant**: Best for everyday questions
   - **Microsoft Assistant**: Specialized for Microsoft products
   - **Technical Support**: Optimized for troubleshooting
   - **Creative Writer**: Perfect for creative writing tasks

## 📊 Monitoring

- Monitor API usage in your OpenAI dashboard
- Check application logs in Render dashboard
- Use the stats panel to track conversation metrics

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available under the MIT License.

---

**Need help?** Check the troubleshooting section or create an issue in the repository.
