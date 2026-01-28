"""
FastAPI backend for Data Copilot.

Exposes the graph execution as a streaming endpoint (Server-Sent Events)
to allow real-time token generation on the UI.
"""
import json
import asyncio
from typing import Optional, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from data_copilot.config import settings
from data_copilot.agents.graph import create_graph, DataCopilotGraph
from data_copilot.observability.mlflow_integration import TracedDataCopilot, setup_mlflow
from data_copilot.tools.sql_executor import get_executor


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    stream: bool = Field(False, description="Whether to stream the response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the loss ratio for class code 8810 in California?",
                "thread_id": "session_123",
                "stream": True
            }
        }


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str
    response: str
    intent: Optional[str] = None
    sql: Optional[str] = None
    execution_time_ms: float
    row_count: int = 0
    thread_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_connected: bool
    mlflow_connected: bool
    timestamp: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_runs: int
    runs_with_reflection: int
    reflection_rate: float
    avg_retries_per_run: float


# Global instances
_graph: Optional[DataCopilotGraph] = None
_traced_copilot: Optional[TracedDataCopilot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _graph, _traced_copilot
    
    logger.info("Starting Data Copilot API...")
    
    # Initialize components
    setup_mlflow()
    _graph = create_graph()
    _traced_copilot = TracedDataCopilot(graph=_graph)
    
    logger.info("Data Copilot API started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Data Copilot API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Data Copilot API",
        description="Natural language interface for insurance data queries",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check database
    executor = get_executor()
    db_connected, db_message = executor.test_connection()
    
    # Check MLFlow
    try:
        import mlflow
        mlflow_connected = mlflow.get_tracking_uri() is not None
    except:
        mlflow_connected = False
    
    status = "healthy" if db_connected else "degraded"
    
    return HealthResponse(
        status=status,
        database_connected=db_connected,
        mlflow_connected=mlflow_connected,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a natural language query.
    
    Returns the complete response after processing.
    """
    global _traced_copilot
    
    if _traced_copilot is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = datetime.now()
    
    try:
        result = _traced_copilot.invoke(
            query=request.query,
            thread_id=request.thread_id,
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract SQL result info
        sql_result = result.get("sql_result", {})
        
        return QueryResponse(
            query=request.query,
            response=result.get("final_response", "No response generated"),
            intent=result.get("intent").value if result.get("intent") else None,
            sql=result.get("generated_sql"),
            execution_time_ms=execution_time,
            row_count=sql_result.get("row_count", 0) if sql_result else 0,
            thread_id=request.thread_id or "default",
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Process a query with streaming response.
    
    Returns Server-Sent Events (SSE) for real-time updates.
    """
    global _graph
    
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate SSE events from graph execution."""
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Stream graph execution
            for event in _graph.stream(request.query, thread_id=request.thread_id):
                for node_name, node_output in event.items():
                    # Send node update
                    event_data = {
                        "type": "node_update",
                        "node": node_name,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    # Add relevant data based on node type
                    if node_name == "router":
                        intent = node_output.get("intent")
                        event_data["intent"] = intent.value if hasattr(intent, 'value') else str(intent)
                    elif node_name == "nl2sql":
                        event_data["sql"] = node_output.get("generated_sql", "")[:500]
                    elif node_name == "executor":
                        sql_result = node_output.get("sql_result", {})
                        event_data["row_count"] = sql_result.get("row_count", 0) if sql_result else 0
                        event_data["success"] = sql_result.get("success", False) if sql_result else False
                    elif node_name == "generator":
                        event_data["response"] = node_output.get("final_response", "")
                    
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Small delay for readability
                    await asyncio.sleep(0.01)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'timestamp': datetime.now().isoformat()})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get aggregated metrics from MLFlow."""
    from data_copilot.observability.mlflow_integration import log_reflection_metrics
    
    try:
        metrics = log_reflection_metrics()
        return MetricsResponse(
            total_runs=metrics.get("total_runs", 0),
            runs_with_reflection=metrics.get("runs_with_reflection", 0),
            reflection_rate=metrics.get("reflection_rate", 0.0),
            avg_retries_per_run=metrics.get("avg_retries_per_run", 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
async def get_schema():
    """Get the data schema information."""
    from data_copilot.data.schema import InsuranceSchema, SCHEMA_DESCRIPTION
    
    return {
        "description": SCHEMA_DESCRIPTION,
        "columns": [
            {
                "name": col.name,
                "type": col.dtype.value,
                "description": col.description,
                "sample_values": col.sample_values,
                "is_metric": col.is_metric,
                "is_dimension": col.is_dimension,
            }
            for col in InsuranceSchema.all_columns()
        ]
    }


@app.get("/history/{thread_id}")
async def get_conversation_history(thread_id: str):
    """Get conversation history for a thread."""
    global _graph
    
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        state = _graph.get_state(thread_id)
        if state:
            return {
                "thread_id": thread_id,
                "messages": state.values.get("messages", []),
            }
        else:
            return {
                "thread_id": thread_id,
                "messages": [],
            }
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "data_copilot.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run_server()
