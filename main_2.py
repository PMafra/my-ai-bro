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
import spacy
import re


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


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm"
    )


app = FastAPI(title="Vector DB Enhanced AI Chatbot", lifespan=lifespan)


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


def parse_relationships(text: str) -> dict:
    """
    Very naive, rule-based extraction of user, father, dog.
    Returns a dict with keys: user_name, father_name, dog_name
    E.g.:
      {
        "user_name": "Pedro",
        "father_name": "Marcelo",
        "dog_name": "Pucky"
      }
    """
    rel_data = {
        "user_name": None,
        "father_name": None,
        "dog_name": None
    }

    # Pattern to detect "I'm <name>"
    match_user = re.search(r"\bI(?:\'m|\s+am)\s+([A-Z][a-zA-Z]+)", text)
    if match_user:
        rel_data["user_name"] = match_user.group(1)

    # Pattern to detect "my father is <name>"
    match_father = re.search(r"\bmy father is\s+([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
    if match_father:
        rel_data["father_name"] = match_father.group(1)

    # Pattern to detect "my dog is <name>"
    match_dog = re.search(r"\bmy dog is\s+([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
    if match_dog:
        rel_data["dog_name"] = match_dog.group(1)

    return rel_data


def store_family_relationships(rel_data: dict):
    """
    Create a (Person {name: user_name}) node if missing.
    If father_name is present, create (father:Person {name: father_name}) node
      and (user)-[:HAS_FATHER]->(father).
    If dog_name is present, create (dog:Dog {name: dog_name}) node
      and (user)-[:HAS_DOG]->(dog).
    """
    user = rel_data.get("user_name")
    father = rel_data.get("father_name")
    dog = rel_data.get("dog_name")

    if not user and not father and not dog:
        return  # nothing to store

    with driver.session() as session:
        session.execute_write(_store_rel_tx, user, father, dog)


def _store_rel_tx(tx, user_name, father_name, dog_name):
    """
    Build a Cypher query dynamically depending on what's present.
    This approach MERGEs the nodes/relationships so we don't create duplicates.
    """
    # We'll do MERGE statements for each relationship found.
    # We separate them for clarity, though you could chain them in one big query.
    if user_name:
        # Merge user as a Person node
        tx.run("""
            MERGE (u:Person {name: $user_name})
        """, user_name=user_name)

    if father_name and user_name:
        tx.run("""
            MERGE (u:Person {name: $user_name})
            MERGE (f:Person {name: $father_name})
            MERGE (u)-[:HAS_FATHER]->(f)
        """, user_name=user_name, father_name=father_name)

    if dog_name and user_name:
        tx.run("""
            MERGE (u:Person {name: $user_name})
            MERGE (d:Dog {name: $dog_name})
            MERGE (u)-[:HAS_DOG]->(d)
        """, user_name=user_name, dog_name=dog_name)


# --------------------------
#   NER (for other mentions)
# --------------------------
def extract_spacy_entities(text: str):
    """
    We'll still do a general SpaCy NER pass, in case we want
    to store other mentions that don't fit father/dog patterns.
    Return a list of {entity, type}.
    """
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({"entity": ent.text, "type": ent.label_})
    return ents

def store_generic_mentions(session_id: str, entities: List[dict]):
    """
    If we want to store leftover mentions, we can keep the old approach
    (Session)-[:MENTIONED]->(Entity).
    We'll skip father/dog from here to avoid duplicating them.
    """
    with driver.session() as neo_session:
        for ent in entities:
            name = ent["entity"]
            e_type = ent["type"]
            # We skip father/dog logic because it's handled above
            # But let's keep everything else
            neo_session.execute_write(_store_mention, session_id, name, e_type)

def _store_mention(tx, session_id, entity_name, entity_type):
    query = """
    MERGE (s:Session {id: $session_id})
    MERGE (e:Entity {name: $entity_name, type: $entity_type})
    MERGE (s)-[:MENTIONED]->(e)
    """
    tx.run(query,
        session_id=session_id,
        entity_name=entity_name,
        entity_type=entity_type
    )


@app.post("/chat-vector")
def chat_endpoint(user_message: UserMessage):
    session_id = user_message.session_id.strip()
    user_text = user_message.message.strip()

    # 1) Maintain local conversation
    conversation_history.setdefault(session_id, []).append(f"User: {user_text}")

    # 2) Parse father/dog relationships from the text
    rel_data = parse_relationships(user_text)
    store_family_relationships(rel_data)

    # 3) Use SpaCy NER to store other mentions
    all_entities = extract_spacy_entities(user_text)
    # We'll filter out father/dog references by name if we want to avoid duplicates:
    leftover_ents = []
    for ent in all_entities:
        # If it's the father_name or dog_name, skip. 
        # If it's the user_name, skip as well. 
        if ent["entity"] not in (rel_data["user_name"], rel_data["father_name"], rel_data["dog_name"]):
            leftover_ents.append(ent)
    store_generic_mentions(session_id, leftover_ents)

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
        print("facts", facts)
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

    # for msg in relevant_messages:
    #     messages.append(msg)

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
