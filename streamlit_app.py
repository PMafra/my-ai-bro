import streamlit as st
import requests

# Set the URL to your FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/chat-dynamic-graph"

# Maintain the chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Dynamic Neo4j Chat")

"""
A simple Streamlit interface to interact with the FastAPI endpoint.
Type your message in the box below and click 'Send' to see how it gets
converted into a Cypher query and returned back to you.
"""

user_input = st.text_input("Your message:", key="user_input_text")

if st.button("Send"):
    if user_input.strip():
        # Prepare the data to send to the backend
        payload = {
            "session_id": "my_session",  # or any unique identifier
            "message": user_input.strip()
        }

        try:
            response = requests.post(FASTAPI_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            # Add user's message and the resulting output to chat history
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Cypher**: {data.get('cypher', 'N/A')}\n\n**Result**: {data.get('result', 'No result')}"}
            )

        except requests.exceptions.RequestException as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {str(e)}"}
            )

# Display the conversation
for chat in st.session_state.messages:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        # Assistant / backend response
        st.markdown(f"**Server:** {chat['content']}")
