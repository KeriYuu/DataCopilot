"""
Streamlit frontend for Data Copilot.

A chat interface for querying insurance data using natural language.
"""
import json
import time
from typing import Optional, List, Dict
from datetime import datetime

import streamlit as st
import requests
from sseclient import SSEClient

# Configuration
API_BASE_URL = "http://localhost:8080"


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "show_sql" not in st.session_state:
        st.session_state.show_sql = False
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = False


def render_sidebar():
    """Render the sidebar with settings and info."""
    st.sidebar.title("⚙️ Settings")
    
    # Thread ID
    st.sidebar.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        key="thread_input",
        on_change=lambda: setattr(st.session_state, 'thread_id', st.session_state.thread_input)
    )
    
    # Display options
    st.session_state.show_sql = st.sidebar.checkbox("Show SQL", value=st.session_state.show_sql)
    st.session_state.show_metrics = st.sidebar.checkbox("Show Metrics", value=st.session_state.show_metrics)
    
    # Clear conversation
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.rerun()
    
    st.sidebar.divider()
    
    # Schema info
    st.sidebar.subheader("📊 Available Data")
    with st.sidebar.expander("View Schema"):
        try:
            response = requests.get(f"{API_BASE_URL}/schema", timeout=5)
            if response.status_code == 200:
                schema = response.json()
                
                st.markdown("**Dimensions:**")
                for col in schema["columns"]:
                    if col["is_dimension"]:
                        st.markdown(f"- `{col['name']}`: {col['description'][:50]}...")
                
                st.markdown("**Metrics:**")
                for col in schema["columns"]:
                    if col["is_metric"]:
                        st.markdown(f"- `{col['name']}`: {col['description'][:50]}...")
        except:
            st.warning("Unable to load schema")
    
    # Example queries
    st.sidebar.subheader("💡 Example Queries")
    examples = [
        "What is the loss ratio for class code 8810 in California?",
        "Show me total premium by state for 2023",
        "Which class codes have the highest claim frequency?",
        "What is the average severity in the Northeast region?",
        "What is class code 8810?",
    ]
    
    for example in examples:
        if st.sidebar.button(example[:40] + "...", key=f"ex_{hash(example)}"):
            st.session_state.example_query = example


def render_chat_messages():
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show SQL if available and enabled
            if message.get("sql") and st.session_state.show_sql:
                with st.expander("🔍 SQL Query"):
                    st.code(message["sql"], language="sql")
            
            # Show metrics if available and enabled
            if message.get("metrics") and st.session_state.show_metrics:
                with st.expander("📈 Metrics"):
                    cols = st.columns(3)
                    cols[0].metric("Rows", message["metrics"].get("row_count", 0))
                    cols[1].metric("Time (ms)", f"{message['metrics'].get('execution_time_ms', 0):.0f}")
                    cols[2].metric("Intent", message["metrics"].get("intent", "N/A"))


def query_api(query: str, stream: bool = True) -> Dict:
    """
    Send a query to the API.
    
    Args:
        query: User query
        stream: Whether to use streaming
        
    Returns:
        Response data
    """
    if stream:
        return query_api_stream(query)
    else:
        return query_api_sync(query)


def query_api_sync(query: str) -> Dict:
    """Send a synchronous query to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "thread_id": st.session_state.thread_id,
                "stream": False,
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Unable to connect to API"}
    except Exception as e:
        return {"error": str(e)}


def query_api_stream(query: str) -> Dict:
    """Send a streaming query to the API."""
    result = {
        "response": "",
        "sql": None,
        "intent": None,
        "row_count": 0,
        "execution_time_ms": 0,
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={
                "query": query,
                "thread_id": st.session_state.thread_id,
            },
            stream=True,
            timeout=60,
        )
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        
        # Parse SSE events
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    
                    if data.get("type") == "node_update":
                        node = data.get("node")
                        
                        if node == "router":
                            result["intent"] = data.get("intent")
                        elif node == "nl2sql":
                            result["sql"] = data.get("sql")
                        elif node == "executor":
                            result["row_count"] = data.get("row_count", 0)
                        elif node == "generator":
                            result["response"] = data.get("response", "")
                    
                    elif data.get("type") == "error":
                        result["error"] = data.get("message")
                        break
        
        return result
        
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Unable to connect to API. Make sure the backend is running."}
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Data Copilot",
        page_icon="📊",
        layout="wide",
    )
    
    init_session_state()
    
    # Header
    st.title("📊 Data Copilot")
    st.markdown("*Query insurance data using natural language*")
    
    # Sidebar
    render_sidebar()
    
    # Check for example query
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        del st.session_state.example_query
        process_query(query)
    
    # Chat interface
    render_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the insurance data..."):
        process_query(prompt)


def process_query(query: str):
    """Process a user query and display results."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            start_time = time.time()
            result = query_api(query, stream=False)  # Use sync for simplicity
            elapsed = (time.time() - start_time) * 1000
        
        if "error" in result:
            st.error(f"❌ {result['error']}")
            response_content = f"Sorry, I encountered an error: {result['error']}"
        else:
            response_content = result.get("response", "No response generated")
            st.markdown(response_content)
            
            # Show SQL if available
            if result.get("sql") and st.session_state.show_sql:
                with st.expander("🔍 SQL Query"):
                    st.code(result["sql"], language="sql")
            
            # Show metrics
            if st.session_state.show_metrics:
                cols = st.columns(4)
                cols[0].metric("Rows", result.get("row_count", 0))
                cols[1].metric("Time (ms)", f"{result.get('execution_time_ms', elapsed):.0f}")
                cols[2].metric("Intent", result.get("intent", "N/A"))
                cols[3].metric("Thread", st.session_state.thread_id[:12] + "...")
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "sql": result.get("sql"),
        "metrics": {
            "row_count": result.get("row_count", 0),
            "execution_time_ms": result.get("execution_time_ms", elapsed),
            "intent": result.get("intent"),
        }
    })


if __name__ == "__main__":
    main()
