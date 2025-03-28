import os
from typing import Dict, List
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI instance
app = FastAPI(title="Simple AI Chatbot")

# In-memory storage for conversations
conversation_history: Dict[str, List[str]] = {}

class UserMessage(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat_endpoint(user_message: UserMessage):
    """
    Endpoint to receive a user's message, store context, and return an AI-generated response.
    """
    session_id = user_message.session_id
    message = user_message.message

    # Ensure the session history is initialized
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Append user message to conversation history
    conversation_history[session_id].append(f"User: {message}")

    # Prepare messages for chat completion
    messages = []
    for entry in conversation_history[session_id]:
        if entry.startswith("User:"):
            messages.append({"role": "user", "content": entry[len("User: "):]})
        elif entry.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": entry[len("Assistant: "):]})

    try:
        # Call OpenAI ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stop=["User:", "Assistant:"]
        )

        bot_reply = response.choices[0].message.content.strip()

        # Store the assistant's reply
        conversation_history[session_id].append(f"Assistant: {bot_reply}")

        return {"reply": bot_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
