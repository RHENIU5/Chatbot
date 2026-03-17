"""
AI Chatbot Web Application
Backend: Flask
AI: Ollama (Local LLM)
"""

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client with Ollama API
# Ollama provides an OpenAI-compatible API endpoint
endpoint = os.getenv("ollama_endpoint", "http://localhost:11434")
model = os.getenv("ollama_model", "llama2")

client = OpenAI(
    base_url=endpoint + "/v1",
    api_key="ollama"
)

# Conversation history stored in memory (session-based)
# In production, use database for persistence
conversation_history = []

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are a helpful AI assistant. When providing code, use markdown formatting with triple backticks (```language) for code blocks. 
Use proper formatting:
- Use **bold** for emphasis
- Use `code` for inline code
- Use numbered lists and bullet points
- Use line breaks between paragraphs
- When showing multiple sections, separate them with clear headings (##)
Always format your responses for readability and professional presentation."""

@app.route("/")
def index():
    """
    Home route - renders the main chatbot interface
    """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint - receives user message and returns AI response
    Expects: JSON {message: "user message"}
    Returns: JSON {reply: "ai response"}
    """
    try:
        # Get the JSON data from request
        data = request.get_json()
        
        # Validate that message exists
        if not data or "message" not in data:
            return jsonify({"error": "Message field required"}), 400
        
        user_message = data["message"].strip()
        
        # Validate message is not empty
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages list for API call
        # Include system prompt as first message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history)
        
        # Call Ollama API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract assistant response
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Return response as JSON
        return jsonify({"reply": assistant_message})
    
    except Exception as e:
        # Proper error handling
        error_str = str(e)
        print(f"Error in /chat endpoint: {error_str}")
        
        # Handle authentication errors
        if "401" in error_str or "Unauthorized" in error_str or "User not found" in error_str:
            error_message = "Authentication Error: Invalid or expired API key. Please check your OpenRouter API key in .env file."
        elif "429" in error_str:
            error_message = "Rate Limit Error: Too many requests. Please wait a moment and try again."
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            error_message = "Server Error: OpenRouter API is temporarily unavailable. Please try again later."
        else:
            error_message = f"Error: {error_str}"
        
        return jsonify({"error": error_message}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    """
    Clear chat history endpoint
    """
    global conversation_history
    conversation_history = []
    return jsonify({"status": "Chat cleared"})

if __name__ == "__main__":
    # Run the Flask development server on port 8000
    # (Port 5000 may be in use by AirPlay Receiver on macOS)
    app.run(debug=True, host="127.0.0.1", port=8000)
