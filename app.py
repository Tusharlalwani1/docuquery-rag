import streamlit as st
import requests

st.set_page_config(page_title="DocuQuery RAG", page_icon="🤖", layout="centered")
st.title("DocuQuery RAG Engine 🤖")
st.caption("A local Retrieval-Augmented Generation system powered by FastAPI.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input handler
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_answer = ""
        
        try:
            # Stream the response from your FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={"query": user_query},
                stream=True
            )
            response.raise_for_status()
            
            # Render text chunks as they arrive for the typewriter effect
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    full_answer += chunk
                    response_placeholder.markdown(full_answer + "▌")
            
            response_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend. Is uvicorn running?")