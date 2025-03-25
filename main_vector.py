import os
import uuid
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import openai
from pinecone import Pinecone, ServerlessSpec

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


INDEX_NAME = "simple-chat-index"

# if INDEX_NAME not in pc.list_indexes():
#     pc.create_index(name=INDEX_NAME, dimension=1536, spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(INDEX_NAME)

app = FastAPI(title="Vector DB Enhanced AI Chatbot")

conversation_history: Dict[str, List[str]] = {}

class UserMessage(BaseModel):
    session_id: str
    message: str

@app.on_event("startup")
def startup_event():
    """
    This runs automatically when the server starts.
    """
    pass

@app.post("/chat-vector")
def chat_endpoint(user_message: UserMessage):
    """
    Endpoint to receive a user's message, store context, 
    retrieve relevant data from Pinecone, and return an AI-generated response.
    """
    session_id = user_message.session_id
    user_text = user_message.message.strip()

    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append(f"User: {user_text}")

    try:
        embedding_response = openai.embeddings.create(
            input=user_text,
            model="text-embedding-3-small"
        )
        user_vector = embedding_response.data[0].embedding

        doc_id = f"{session_id}-{str(uuid.uuid4())}"

        index.upsert(
            vectors=[
                (
                    doc_id, 
                    user_vector, 
                    {"role": "user", "text": user_text, "session": session_id}
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating/upserting embedding: {str(e)}")

    # Retrieve top similar messages from Pinecone for additional context
    try:
        search_result = index.query(
            vector=user_vector,
            top_k=3,  # get top 3 semantically similar
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

    relevant_messages = []
    for match in search_result["matches"]:
        metadata = match["metadata"]
        if metadata.get("session") == session_id:
            text = metadata.get("text", "")
            role = metadata.get("role", "user")
            if role == "user":
                relevant_messages.append({"role": "user", "content": text})
            else:
                relevant_messages.append({"role": "assistant", "content": text})

    # Sort these by semantic similarity (not strictly needed if top_k is small)
    # but let's assume the result is already sorted by Pinecone.

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide clear and concise answers."
        }
    ]

    for msg in relevant_messages:
        messages.append(msg)

    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stop=["User:", "Assistant:"]
        )
        bot_reply = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat completion: {str(e)}")

    conversation_history[session_id].append(f"Assistant: {bot_reply}")

    try:
        embedding_response = openai.embeddings.create(
            input=bot_reply,
            model="text-embedding-3-small"
        )
        assistant_vector = embedding_response.data[0].embedding
        doc_id = f"{session_id}-{str(uuid.uuid4())}"

        index.upsert(
            vectors=[
                (
                    doc_id,
                    assistant_vector,
                    {"role": "assistant", "text": bot_reply, "session": session_id}
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing assistant reply: {str(e)}")

    return {"reply": bot_reply}
