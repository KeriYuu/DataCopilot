"""
MLFlow integration for Data Copilot observability.

Provides full tracing of LangGraph execution, including:
- Node-level spans for Router, Keyword Rewriter, NL2SQL Generator
- Reflection tracking for monitoring retry behavior
- Performance metrics and token usage
"""
import os
import time
from typing import Optional, Dict, Any, Generator
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
from loguru import logger

import mlflow
from mlflow.tracking import MlflowClient

from data_copilot.config import settings
from data_copilot.agents.graph import DataCopilotGraph, create_graph
from data_copilot.agents.state import AgentState, create_initial_state


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    enable_autolog: bool = True,
) -> MlflowClient:
    """
    Initialize MLFlow tracking.
    
    Args:
        tracking_uri: MLFlow tracking server URI
        experiment_name: Experiment name
        enable_autolog: Whether to enable LangChain autologging
        
    Returns:
        MLFlow client instance
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    exp_name = experiment_name or settings.mlflow_experiment_name
    
    mlflow.set_tracking_uri(uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
        logger.info(f"Created MLFlow experiment: {exp_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLFlow experiment: {exp_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(exp_name)
    
    # Enable LangChain autologging if available
    if enable_autolog:
        try:
            mlflow.langchain.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=False,  # Don't log model artifacts
                log_traces=True,
            )
            logger.info("MLFlow LangChain autologging enabled")
        except AttributeError:
            logger.warning("MLFlow LangChain autologging not available (requires mlflow>=2.10)")
    
    return MlflowClient(tracking_uri=uri)


def get_mlflow_client() -> MlflowClient:
    """Get the MLFlow client."""
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


@contextmanager
def trace_node(node_name: str, parent_run_id: Optional[str] = None):
    """
    Context manager for tracing a graph node.
    
    Args:
        node_name: Name of the node being traced
        parent_run_id: Optional parent run ID for nesting
        
    Yields:
        MLFlow run context
    """
    start_time = time.time()
    
    with mlflow.start_run(run_name=node_name, nested=True) as run:
        mlflow.set_tag("node_type", node_name)
        mlflow.set_tag("start_time", datetime.now().isoformat())
        
        try:
            yield run
            
            # Log success
            mlflow.set_tag("status", "success")
            
        except Exception as e:
            # Log failure
            mlflow.set_tag("status", "error")
            mlflow.set_tag("error_message", str(e))
            raise
            
        finally:
            # Log duration
            duration_ms = (time.time() - start_time) * 1000
            mlflow.log_metric("duration_ms", duration_ms)


class TracedDataCopilot:
    """
    MLFlow-traced wrapper for DataCopilotGraph.
    
    Automatically logs:
    - Each node execution as a separate span
    - Reflection/retry steps
    - Final response quality metrics
    - Token usage and latency
    """
    
    def __init__(
        self,
        graph: Optional[DataCopilotGraph] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the traced Data Copilot.
        
        Args:
            graph: Optional pre-configured graph instance
            experiment_name: MLFlow experiment name
        """
        self.graph = graph or create_graph()
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        
        # Setup MLFlow
        setup_mlflow(experiment_name=self.experiment_name)
        
    def invoke(
        self, 
        query: str,
        thread_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a query with full MLFlow tracing.
        
        Args:
            query: User's natural language query
            thread_id: Optional thread ID
            run_name: Optional custom run name
            tags: Optional additional tags
            **kwargs: Additional configuration
            
        Returns:
            Final state with response
        """
        run_name = run_name or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log input
            mlflow.log_param("query", query[:500])  # Truncate for MLFlow limits
            mlflow.log_param("thread_id", thread_id or "default")
            
            # Add custom tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            start_time = time.time()
            
            try:
                # Execute with streaming to capture node-level metrics
                result = self._invoke_with_tracing(query, thread_id, **kwargs)
                
                # Log success metrics
                self._log_success_metrics(result, start_time)
                
                return result
                
            except Exception as e:
                # Log failure
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error_message", str(e))
                mlflow.log_metric("success", 0)
                raise
    
    def _invoke_with_tracing(
        self,
        query: str,
        thread_id: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute graph with node-level tracing.
        
        Args:
            query: User query
            thread_id: Thread ID
            **kwargs: Additional config
            
        Returns:
            Final state
        """
        final_state = None
        node_timings = {}
        
        # Stream to capture intermediate states
        for event in self.graph.stream(query, thread_id=thread_id, **kwargs):
            for node_name, node_output in event.items():
                # Log node execution
                with trace_node(node_name):
                    # Log node-specific metrics
                    self._log_node_metrics(node_name, node_output)
                
                # Track timing
                if node_name not in node_timings:
                    node_timings[node_name] = {"start": time.time()}
                else:
                    node_timings[node_name]["end"] = time.time()
                
                # Update final state
                final_state = node_output if isinstance(node_output, dict) else final_state
        
        # If streaming didn't work, use direct invoke
        if final_state is None:
            final_state = self.graph.invoke(query, thread_id=thread_id, **kwargs)
        
        return final_state
    
    def _log_node_metrics(self, node_name: str, node_output: Dict[str, Any]) -> None:
        """Log metrics specific to each node type."""
        
        if node_name == "router":
            intent = node_output.get("intent")
            if intent:
                mlflow.log_param("intent", intent.value if hasattr(intent, 'value') else str(intent))
            
        elif node_name == "entity_extractor":
            entities = node_output.get("extracted_entities", {})
            mlflow.log_metric("entity_count", len(entities))
            if entities:
                mlflow.log_param("entities", str(entities)[:500])
                
        elif node_name == "nl2sql":
            sql = node_output.get("generated_sql", "")
            if sql:
                mlflow.log_param("generated_sql", sql[:1000])
                mlflow.log_metric("sql_length", len(sql))
                
        elif node_name == "executor":
            result = node_output.get("sql_result", {})
            if result:
                mlflow.log_metric("result_row_count", result.get("row_count", 0))
                mlflow.log_metric("execution_time_ms", result.get("execution_time_ms", 0))
                mlflow.log_metric("sql_success", 1 if result.get("success") else 0)
            error = node_output.get("execution_error")
            if error:
                mlflow.set_tag("execution_error", error[:200])
                
        elif node_name == "reflection":
            retry_count = node_output.get("retry_count", 0)
            mlflow.log_metric("retry_count", retry_count)
            mlflow.set_tag("reflection_triggered", "true")
            
        elif node_name == "generator":
            response = node_output.get("final_response", "")
            mlflow.log_metric("response_length", len(response))
    
    def _log_success_metrics(
        self, 
        result: Dict[str, Any], 
        start_time: float
    ) -> None:
        """Log final success metrics."""
        
        total_time_ms = (time.time() - start_time) * 1000
        
        mlflow.log_metric("total_time_ms", total_time_ms)
        mlflow.log_metric("success", 1)
        mlflow.set_tag("status", "success")
        
        # Log reflection metrics
        reflection_steps = result.get("reflection_steps", [])
        mlflow.log_metric("total_reflections", len(reflection_steps))
        
        # Log token usage if available
        total_tokens = result.get("total_tokens", 0)
        if total_tokens:
            mlflow.log_metric("total_tokens", total_tokens)
        
        # Log final response
        response = result.get("final_response", "")
        if response:
            mlflow.log_text(response, "response.txt")
    
    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific run.
        
        Args:
            run_id: MLFlow run ID
            
        Returns:
            Dictionary of metrics
        """
        client = get_mlflow_client()
        run = client.get_run(run_id)
        
        return {
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "status": run.info.status,
            "duration_ms": run.info.end_time - run.info.start_time if run.info.end_time else None,
        }


def log_reflection_metrics(experiment_name: Optional[str] = None) -> Dict[str, float]:
    """
    Aggregate reflection metrics across runs.
    
    This is useful for monitoring how often the Generator model
    rejects SQL output and initiates self-correction retries.
    
    Args:
        experiment_name: Experiment to analyze
        
    Returns:
        Aggregated metrics
    """
    exp_name = experiment_name or settings.mlflow_experiment_name
    client = get_mlflow_client()
    
    experiment = client.get_experiment_by_name(exp_name)
    if not experiment:
        return {}
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=1000,
    )
    
    total_runs = len(runs)
    reflection_runs = sum(1 for r in runs if r.data.tags.get("reflection_triggered") == "true")
    total_reflections = sum(r.data.metrics.get("total_reflections", 0) for r in runs)
    avg_retries = total_reflections / total_runs if total_runs > 0 else 0
    
    return {
        "total_runs": total_runs,
        "runs_with_reflection": reflection_runs,
        "reflection_rate": reflection_runs / total_runs if total_runs > 0 else 0,
        "total_reflections": total_reflections,
        "avg_retries_per_run": avg_retries,
    }
