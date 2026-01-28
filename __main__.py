"""
Data Copilot main entry point.

Usage:
    python -m data_copilot --help
    python -m data_copilot serve           # Start API server
    python -m data_copilot ui              # Start Streamlit UI
    python -m data_copilot query "..."     # Run a single query
"""
import argparse
import sys
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Data Copilot - Natural language interface for insurance data"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the Streamlit UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--trace", action="store_true", help="Enable MLFlow tracing")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Print the data schema")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        start_server(args)
    elif args.command == "ui":
        start_ui(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "schema":
        print_schema()
    else:
        parser.print_help()


def start_server(args):
    """Start the FastAPI server."""
    import uvicorn
    
    logger.info(f"Starting Data Copilot API on {args.host}:{args.port}")
    
    uvicorn.run(
        "data_copilot.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def start_ui(args):
    """Start the Streamlit UI."""
    import subprocess
    
    logger.info(f"Starting Data Copilot UI on port {args.port}")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "data_copilot/ui/app.py",
        "--server.port", str(args.port),
    ])


def run_query(args):
    """Run a single query."""
    from data_copilot.agents.graph import create_graph
    from data_copilot.observability import TracedDataCopilot
    
    logger.info(f"Processing query: {args.text}")
    
    if args.trace:
        copilot = TracedDataCopilot()
        result = copilot.invoke(args.text)
    else:
        graph = create_graph()
        result = graph.invoke(args.text)
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(result.get("final_response", "No response generated"))
    
    if result.get("generated_sql"):
        print("\n" + "-" * 60)
        print("SQL:")
        print("-" * 60)
        print(result["generated_sql"])


def print_schema():
    """Print the data schema."""
    from data_copilot.data.schema import SCHEMA_DESCRIPTION
    print(SCHEMA_DESCRIPTION)


if __name__ == "__main__":
    main()
