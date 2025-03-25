import os
import uuid
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import openai
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase, basic_auth
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv()

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

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
)


conversation_history: Dict[str, List[str]] = {}

class UserMessage(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App starting up")
    yield
    print("App shutting down")


app = FastAPI(title="Vector DB Enhanced AI Chatbot", lifespan=lifespan)


def extract_entities_and_relationships(text: str) -> List[dict]:
    """
    Naive placeholder for an entity extraction step.
    In real-world usage, you might call an LLM or spaCy NER to parse 'text'
    and return discovered entities/relationships.

    Return format (example):
    [
      {"type": "PERSON", "entity": "Alice"},
      {"type": "LOCATION", "entity": "Paris"}
    ]
    """
    # Just a trivial example:
    entities = []
    if "dog" in text.lower():
        entities.append({"type": "ANIMAL", "entity": "dog"})
        print("ENTITIES", entities)
    return entities


def create_entity_node(tx, session_id: str, entity_info: dict):
    """
    Write transaction for Neo4j. Creates (Entity) node and
    (Session)-[:MENTIONED]->(Entity) relationship, for example.
    """
    query = """
    MERGE (s:Session {id: $session_id})
    MERGE (e:Entity {name: $entity_name, type: $entity_type})
    MERGE (s)-[:MENTIONED]->(e)
    """
    tx.run(query, 
        session_id=session_id,
        entity_name=entity_info["entity"],
        entity_type=entity_info["type"]
    )


def store_entities_in_neo4j(session_id: str, entities: List[dict]):
    """
    For each entity, create a node in Neo4j (if it doesn't exist),
    and associate it with the user session as a relationship.

    This is extremely simplified. Real usage might store the user as well,
    or connect multiple entities with relationships.
    """
    with driver.session() as neo_session:
        for ent in entities:
            new = neo_session.execute_write(create_entity_node, session_id, ent)
            print("CREATION", new)


def get_session_entities(tx, session_id: str):
    """
    Return all entities mentioned by a given session.
    """
    query = """
    MATCH (s:Session {id: $session_id})-[:MENTIONED]->(e:Entity)
    RETURN e.name AS entity_name, e.type AS entity_type
    """
    result = tx.run(query, session_id=session_id)
    return result.data()


def retrieve_relevant_facts(session_id: str) -> List[str]:
    """
    Query Neo4j for facts/relationships that might be relevant
    to the user's session. Then return them as text strings.
    """
    with driver.session() as neo_session:
        records = neo_session.execute_read(get_session_entities, session_id)
    # Convert the returned graph data into a list of textual statements
    facts = []
    for r in records:
        entity_name = r["entity_name"]
        entity_type = r["entity_type"]
        # Example text. In real usage, you might store more properties
        fact_text = f"Session {session_id} mentioned a {entity_type} named {entity_name}."
        facts.append(fact_text)
    return facts


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

    entities = extract_entities_and_relationships(user_text)
    store_entities_in_neo4j(session_id, entities)

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

    try:
        facts = retrieve_relevant_facts(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving facts from Neo4j: {str(e)}")


    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with knowledge of prior context.\n"
                "Below are some facts:\n"
                f"{chr(10).join(facts)}\n"
                "Respond using these facts when relevant."
            )
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
