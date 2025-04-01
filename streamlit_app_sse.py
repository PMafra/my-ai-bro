import streamlit as st
import requests

SSE_ENDPOINT = "http://localhost:8000/chat-dynamic-graph-sse"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Dynamic Neo4j Chat (SSE)")

user_input = st.text_input("Your message:")
send_button = st.button("Send")

def stream_sse_from_fastapi(session_id: str, message: str):
    """
    Streams lines from the SSE endpoint.
    We look for lines starting with 'data:'.
    """
    params = {"session_id": session_id, "message": message}
    with requests.get(SSE_ENDPOINT, params=params, stream=True) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    chunk = line.replace("data:", "").strip()
                    yield chunk

if send_button and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})

    partial_response = ""
    placeholder = st.empty()

    for token in stream_sse_from_fastapi("my_session", user_input.strip()):
        if not token.startswith(" "):
            partial_response += " "
        partial_response += token
        placeholder.markdown(f"**Server:** {partial_response}")

    st.session_state["messages"].append({"role": "assistant", "content": partial_response})

st.write("---")
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Server:** {msg['content']}")
