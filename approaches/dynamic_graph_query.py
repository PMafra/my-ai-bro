from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import openai
from neo4j import GraphDatabase, basic_auth
import os
from openai import OpenAI
from dotenv import load_dotenv

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


@app.post("/chat-dynamic-graph")
def chat_dynamic_graph(user_query: UserQuery):
    user_text = user_query.message.strip()

    try:
        llm_output = translate_to_cypher(user_text)  
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    action = llm_output.get("action")
    cypher = llm_output.get("cypher")

    if not action or not cypher:
        raise HTTPException(status_code=500, detail="LLM output is missing required fields.")

    try:
        with driver.session() as session:
            if action in ["create", "update"]:
                session.run(cypher)
                return {"result": f"Successfully executed {action} query.", "cypher": cypher}
            elif action == "query":
                result = session.run(cypher)
                records = list(result)
                data = []
                for r in records:
                    data.append(dict(r))
                answer = summarize_query_results(user_text, data)
                return {"result": answer, "data": data, "cypher": cypher}
            else:
                return {"result": f"Unknown action: {action}", "cypher": cypher}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cypher execution failed: {str(e)}")

def summarize_query_results(user_text: str, query_results: List[dict]) -> str:
    """
    Optionally feed the results back to the LLM to generate a human-friendly answer.
    For brevity, we do a direct string formatting here.
    """
    if not query_results:
        return "No data found."
    else:
        result_str = str(query_results)
        summary_prompt = f"User asked: '{user_text}'\nData: {result_str}\nPlease summarize the above data in a short answer."
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summarizing query results."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
