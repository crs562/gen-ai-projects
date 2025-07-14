#!/bin/bash

# SQL Chat Agent Startup Script
echo "🚀 Starting SQL Chat Agent..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if database exists, if not create it
if [ ! -f "student.db" ]; then
    echo "🗄️ Creating database..."
    python sqlite.py
else
    echo "✅ Database already exists"
fi

# Check for API key
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY not set as environment variable"
    echo "💡 You can set it in the Streamlit app or export it:"
    echo "   export GROQ_API_KEY='your_api_key_here'"
    echo ""
fi

# Start the application
echo "🌟 Launching Streamlit app..."
echo "📱 The app will open in your browser at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0