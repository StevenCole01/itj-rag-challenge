import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="arXiv RAG Explorer",
    page_icon="📄",
    layout="wide"
)

# App Title
st.title("📄 arXiv Research Assistant")
st.markdown("Ask questions about the research papers stored in your `./data` directory.")

# Sidebar for status and help
with st.sidebar:
    st.header("System Status")
    try:
        # Check backend health
        health_resp = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        if health_resp.status_code == 200:
            st.success("Backend: Connected")
        else:
            st.error("Backend: Error")
    except Exception:
        st.error("Backend: Not Found (Ensure FastAPI is running)")
    
    st.divider()
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=3,
        max_value=10,
        value=5
    )
    st.divider()
    st.markdown("**Instructions**")
    st.write("1. Place PDFs in `./data`, then index them:")
    st.code("python scripts/ingest.py", language="bash")
    st.write("2. Start the API:")
    st.code("python -m app.main", language="bash")
    st.write("3. Start the UI:")
    st.code("streamlit run ui/streamlit_app.py", language="bash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.write(f"**Source:** {source['source']} (Page {source['page']})")
                    st.caption(f"Context: {source['text'][:200]}...")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query through backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/v1/query",
                    json={"query": prompt, "top_k": top_k},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    
                    # Store assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Display sources in a collapsible section
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.write(f"**Source:** {source['source']} (Page {source['page']})")
                            st.caption(f"Context: {source['text'][:200]}...")
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")
