import os
import json
import sqlite3
import csv
from datetime import datetime
import logging
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DATABASE_NAME = "customer_support.db"
CSV_FILE_NAME = "user_data.csv"

# SMTP Configuration
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "support@techeduhub.com")

# Default models
GROQ_MODELS_DEFAULT = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
OLLAMA_MODELS_DEFAULT = ["llama3:latest", "codellama:7b", "gemma:latest", "gemma:2b"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database utilities
def init_db():
    """Initialize SQLite database and create tables"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS support_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT,
                ai_response TEXT,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        conn.commit()
        logging.info(f"Database '{DATABASE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_user_from_db(email):
    """Fetch user by email from database"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at FROM users WHERE email = ?", (email,))
        user_row = cursor.fetchone()
        if user_row:
            return {
                "id": user_row[0], 
                "name": user_row[1], 
                "email": user_row[2], 
                "created_at": user_row[3]
            }
    except sqlite3.Error as e:
        logging.error(f"Error fetching user {email} from DB: {e}")
    finally:
        if conn:
            conn.close()
    return None

def add_or_get_user_in_db(name, email):
    """Add new user or get existing user"""
    user = get_user_from_db(email)
    if user:
        return user
    
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
        conn.commit()
        user_id = cursor.lastrowid
        return {
            "id": user_id, 
            "name": name, 
            "email": email, 
            "created_at": datetime.now().isoformat()
        }
    except sqlite3.IntegrityError:
        return get_user_from_db(email)
    except sqlite3.Error as e:
        logging.error(f"Error adding user {email} to DB: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_user_interaction_history_from_db(email, limit=5):
    """Fetch recent interaction history for user"""
    history = []
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_message, ai_response, timestamp FROM support_interactions
            WHERE user_email = ? ORDER BY timestamp DESC LIMIT ?
        ''', (email, limit))
        rows = cursor.fetchall()
        
        for row in reversed(rows):
            history.append({"role": "user", "content": row[0], "timestamp": row[2]})
            history.append({"role": "assistant", "content": row[1], "timestamp": row[2]})
    except sqlite3.Error as e:
        logging.error(f"Error fetching interaction history for {email}: {e}")
    finally:
        if conn:
            conn.close()
    return history

def add_support_interaction_to_db(user_email, user_message, ai_response):
    """Add support interaction to database"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO support_interactions (user_email, user_message, ai_response)
            VALUES (?, ?, ?)
        ''', (user_email, user_message, ai_response))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error adding interaction for {user_email}: {e}")
    finally:
        if conn:
            conn.close()

def get_user_data_from_csv(email):
    """Fetch additional user data from CSV file"""
    user_csv_data = {}
    try:
        if not os.path.exists(CSV_FILE_NAME):
            return user_csv_data
        
        with open(CSV_FILE_NAME, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('email') == email:
                    user_csv_data = row
                    break
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
    return user_csv_data

# LLM interaction functions
def get_groq_response(messages, model):
    """Get response from Groq API"""
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'messages': messages,
        'model': model,
        'temperature': 0.6,
        'max_tokens': 1024,
        'top_p': 1,
        'stream': False
    }
    
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers,
        json=data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def get_ollama_response(messages, model):
    """Get response from Ollama"""
    data = {
        'model': model,
        'messages': messages,
        'stream': False
    }
    
    response = requests.post(
        f'{OLLAMA_HOST}/api/chat',
        json=data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()['message']['content']

def build_conversation_history(user_details, user_csv_data, past_interactions):
    """Build initial conversation history with context"""
    user_context = f"User: {user_details.get('name', 'Guest')} ({user_details.get('email', 'N/A')})."
    if user_details.get('created_at'):
        user_context += f"\nDB Profile: Joined on {user_details.get('created_at')}."
    
    if user_csv_data:
        csv_info = {k: v for k, v in user_csv_data.items() if k != 'email'}
        if csv_info:
            user_context += f"\nAdditional Info: {json.dumps(csv_info)}."
    
    system_prompt = (
        "Your Name is Bhevin. You are an advanced AI customer support assistant for 'Tech Edu Hub'. "
        "Mission: Empower individuals with practical IT skills through project-based learning. "
        "Be polite, knowledgeable, concise, and helpful. "
        f"Current User Context:\n{user_context}"
    )
    
    history = [{"role": "system", "content": system_prompt}]
    history.extend(past_interactions)
    return history

def send_email_transcript(user_email, user_name, conversation_history):
    """Send email transcript to user"""
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
        raise ValueError("SMTP configuration incomplete")
    
    transcript = f"Chat Transcript with Tech Edu Hub Support\n"
    transcript += f"User: {user_name} ({user_email})\n"
    transcript += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    transcript += "=" * 40 + "\n\n"
    
    for message in conversation_history:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        elif role == "user":
            transcript += f"{user_name}: {content}\n\n"
        elif role == "assistant":
            transcript += f"Bhevin (AI): {content}\n\n"
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = user_email
    msg['Subject'] = "Chat Transcript - Tech Edu Hub Support"
    msg.attach(MIMEText(transcript, 'plain', 'utf-8'))
    
    smtp_port = int(SMTP_PORT)
    if smtp_port == 465:
        server = smtplib.SMTP_SSL(SMTP_HOST, smtp_port)
    else:
        server = smtplib.SMTP(SMTP_HOST, smtp_port)
        server.starttls()
    
    server.login(SMTP_USER, SMTP_PASSWORD)
    server.send_message(msg)
    server.quit()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        data = request.json
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        llm_provider = data.get('llm_provider')
        llm_model = data.get('llm_model')
        
        if not all([name, email, llm_provider, llm_model]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        # Validate LLM configuration
        if llm_provider == 'groq' and not GROQ_API_KEY:
            return jsonify({'success': False, 'error': 'Groq API key not configured'}), 400
        
        # Create or get user
        user_details = add_or_get_user_in_db(name, email)
        if not user_details:
            return jsonify({'success': False, 'error': 'Failed to create/get user'}), 500
        
        # Get user data
        user_csv_data = get_user_data_from_csv(email)
        past_interactions = get_user_interaction_history_from_db(email, limit=3)
        
        # Build conversation history
        conversation_history = build_conversation_history(user_details, user_csv_data, past_interactions)
        
        # Store in session
        session['user_details'] = user_details
        session['user_csv_data'] = user_csv_data
        session['llm_provider'] = llm_provider
        session['llm_model'] = llm_model
        session['conversation_history'] = conversation_history
        
        # Create initial greeting
        greeting = f"Hello {name}! I am Bhevin, your AI assistant from Tech Edu Hub. How can I help you today?"
        if user_csv_data.get('notes'):
            greeting += f"\n\nI see a note here: \"{user_csv_data['notes']}\". Is this related to your query?"
        
        conversation_history.append({"role": "assistant", "content": greeting})
        session['conversation_history'] = conversation_history
        
        return jsonify({
            'success': True, 
            'greeting': greeting,
            'user_info': f"{name} ({email}) | {llm_provider.upper()} ({llm_model})"
        })
        
    except Exception as e:
        logging.error(f"Error starting session: {e}")
        return jsonify({'success': False, 'error': 'Failed to start session'}), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        if 'user_details' not in session:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        # Get session data
        conversation_history = session.get('conversation_history', [])
        llm_provider = session.get('llm_provider')
        llm_model = session.get('llm_model')
        user_details = session.get('user_details')
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Get AI response
        try:
            if llm_provider == 'groq':
                ai_response = get_groq_response(conversation_history, llm_model)
            elif llm_provider == 'ollama':
                ai_response = get_ollama_response(conversation_history, llm_model)
            else:
                raise ValueError("Invalid LLM provider")
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            ai_response = f"I apologize, but I'm having trouble connecting to the AI service right now. Please try again later."
        
        # Add AI response to history
        conversation_history.append({"role": "assistant", "content": ai_response})
        session['conversation_history'] = conversation_history
        
        # Save to database
        add_support_interaction_to_db(
            user_details['email'], 
            user_message, 
            ai_response
        )
        
        return jsonify({
            'success': True,
            'response': ai_response
        })
        
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return jsonify({'success': False, 'error': 'Failed to process message'}), 500

@app.route('/get_models/<provider>')
def get_models(provider):
    """Get available models for specified provider"""
    if provider == 'groq':
        return jsonify({'models': GROQ_MODELS_DEFAULT})
    elif provider == 'ollama':
        # Try to get models from Ollama, fallback to defaults
        try:
            response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
            if response.status_code == 200:
                ollama_models = [model['name'] for model in response.json().get('models', [])]
                return jsonify({'models': ollama_models if ollama_models else OLLAMA_MODELS_DEFAULT})
        except:
            pass
        return jsonify({'models': OLLAMA_MODELS_DEFAULT})
    else:
        return jsonify({'models': []})

@app.route('/email_transcript', methods=['POST'])
def email_transcript():
    try:
        if 'user_details' not in session:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        user_details = session.get('user_details')
        conversation_history = session.get('conversation_history', [])
        
        send_email_transcript(
            user_details['email'],
            user_details['name'],
            conversation_history
        )
        
        return jsonify({'success': True, 'message': 'Transcript sent successfully'})
        
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return jsonify({'success': False, 'error': 'Failed to send email'}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    session.clear()
    return jsonify({'success': True})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    if 'conversation_history' in session:
        # Keep system message and rebuild initial greeting
        user_details = session.get('user_details', {})
        user_csv_data = session.get('user_csv_data', {})
        
        system_message = None
        for msg in session['conversation_history']:
            if msg.get('role') == 'system':
                system_message = msg
                break
        
        conversation_history = [system_message] if system_message else []
        
        # Add fresh greeting
        name = user_details.get('name', 'there')
        greeting = f"Hello {name}! I am Chaitanya, your AI assistant from Tech Edu Hub. How can I help you today?"
        if user_csv_data.get('notes'):
            greeting += f"\n\nI see a note here: \"{user_csv_data['notes']}\". Is this related to your query?"
        
        conversation_history.append({"role": "assistant", "content": greeting})
        session['conversation_history'] = conversation_history
        
        return jsonify({'success': True, 'greeting': greeting})
    
    return jsonify({'success': False, 'error': 'No active session'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))