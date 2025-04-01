from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import openai
from neo4j import GraphDatabase, basic_auth
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from fastapi.responses import StreamingResponse

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(title="Dynamic Neo4j Chat")

class UserQuery(BaseModel):
    session_id: str
    message: str


def translate_to_cypher(user_text: str) -> dict:
    """
    Calls an LLM (e.g., GPT-3.5/4) to convert the user's message into an
    action (create/query/etc.) and a Cypher query. Returns a dict with
    keys `action` and `cypher`.
    """

    system_prompt = """\
You are a translator from plain English to Cypher queries for a Neo4j graph database.

Constraints and instructions:
1. If the user input implies creating or updating data, generate a Cypher statement that merges or updates nodes/relationships accordingly.
2. If the user input is asking a question about data, generate a Cypher statement that queries the graph.
3. Output MUST be valid JSON with exactly two keys: "action" and "cypher".
4. "action" can be: "create", "update", "query", or similar.
5. "cypher" is the query or queries to run. 
6. Do not include any additional keys or text outside of the JSON.

Example:
User: "I'm Pedro and my father is Marcelo."
Response (JSON):
{
  "action": "create",
  "cypher": "MERGE (p:Person {name: 'Pedro'}) MERGE (f:Person {name: 'Marcelo'}) MERGE (p)-[:HAS_FATHER]->(f)"
}

User: "Who is Pedro's father?"
Response (JSON):
{
  "action": "query",
  "cypher": "MATCH (p:Person)-[:HAS_FATHER]->(f:Person) WHERE p.name = 'Pedro' RETURN f.name AS father"
}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.2,
    )

    response_text = completion.choices[0].message.content.strip()

    import json
    try:
        output = json.loads(response_text)
        return output
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON:\n{response_text}")

def sse_stream_llm_answer(prompt: str, query_data: List[dict]):
    """
    We'll feed 'query_data' to GPT in streaming mode to produce
    a user-friendly answer. 
    """
    # Convert data to a string the model can read
    data_str = json.dumps(query_data, indent=2)

    # Build a system + user prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the query results for the user."},
        {
            "role": "user",
            "content": (
                f"User's original prompt: {prompt}\n"
                f"Query results:\n{data_str}\n"
                "Please provide a concise answer."
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    for chunk in response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = delta.content
        if content:
            yield f"data: {content}\n\n"

@app.get("/chat-dynamic-graph-sse")
def chat_dynamic_graph_sse(session_id: str, message: str):
    try:
        output = translate_to_cypher(message)
    except Exception as e:
        def error_gen():
            yield f"data: [ERROR parsing LLM output]: {str(e)}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    action = output.get("action")
    cypher = output.get("cypher")

    if not action or not cypher:
        def error_gen():
            yield "data: [ERROR] Missing 'action' or 'cypher'.\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    with driver.session() as session:
        if action in ["create", "update"]:
            try:
                session.run(cypher)
                def success_stream():
                    yield f"data: Successfully executed {action} query.\n\n"
                return StreamingResponse(success_stream(), media_type="text/event-stream")
            except Exception as e:
                def error_gen():
                    yield f"data: [ERROR executing {action}]: {str(e)}\n\n"
                return StreamingResponse(error_gen(), media_type="text/event-stream")

        elif action == "query":
            try:
                result = session.run(cypher)
                records = list(result)
                data = [record_to_dict(r) for r in records]  # <--- FIX
                print("##########", data)
            except Exception as e:
                def error_gen():
                    yield f"data: [ERROR running query]: {str(e)}\n\n"
                return StreamingResponse(error_gen(), media_type="text/event-stream")

            return StreamingResponse(
                sse_stream_llm_answer(message, data),
                media_type="text/event-stream"
            )
        else:
            def unknown_gen():
                yield f"data: [WARNING] Unknown action '{action}'. Attempting to run Cypher...\n\n"
                try:
                    session.run(cypher)
                    yield f"data: Cypher executed successfully.\n\n"
                except Exception as e2:
                    yield f"data: [ERROR running Cypher]: {str(e2)}\n\n"
            return StreamingResponse(unknown_gen(), media_type="text/event-stream")
        
from neo4j.graph import Node, Relationship

def neo4j_value_to_dict(value):
    """
    Convert a Neo4j Node/Relationship to a dict,
    or pass through basic types (str, int, float, etc.).
    """
    if isinstance(value, Node):
        return {
            "id": value.element_id,
            "labels": list(value.labels),
            "properties": dict(value)
        }
    elif isinstance(value, Relationship):
        return {
            "id": value.element_id,
            "type": value.type,
            "start_id": value.start_node.element_id,
            "end_id": value.end_node.element_id,
            "properties": dict(value)
        }
    else:
        return value

def record_to_dict(record):
    """
    Convert an entire record (which may have multiple keys)
    into a JSON-friendly dict.
    """
    d = {}
    for key, val in record.items():
        d[key] = neo4j_value_to_dict(val)
    return d
