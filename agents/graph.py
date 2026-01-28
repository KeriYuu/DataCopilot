"""
LangGraph state machine definition for Data Copilot.

This module defines the directed acyclic graph (DAG) that orchestrates
the agent's workflow, including conditional routing and reflection loops.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data_copilot.agents.state import AgentState, create_initial_state
from data_copilot.agents.nodes import (
    router_node,
    entity_extractor_node,
    nl2sql_node,
    executor_node,
    reflection_node,
    generator_node,
)
from data_copilot.agents.nodes.router import should_route_to_analysis
from data_copilot.agents.nodes.executor import should_reflect


class DataCopilotGraph:
    """
    LangGraph-based Data Copilot agent.
    
    The pipeline transforms unstructured natural language into precise
    analytical execution through a directed graph with conditional edges
    for routing and reflection.
    
    Graph structure:
        User Query → Router → [Entity Extractor → NL2SQL → Executor → (Reflection loop)] → Generator → Response
                           ↘ (Metadata/General) → Generator → Response
    """
    
    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        """
        Initialize the Data Copilot graph.
        
        Args:
            checkpointer: Optional state checkpointer for persistence
        """
        self.graph = self._build_graph()
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Returns:
            Compiled StateGraph
        """
        # Initialize graph with state schema
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("router", router_node)
        graph.add_node("entity_extractor", entity_extractor_node)
        graph.add_node("nl2sql", nl2sql_node)
        graph.add_node("executor", executor_node)
        graph.add_node("reflection", reflection_node)
        graph.add_node("generator", generator_node)
        
        # Set entry point
        graph.set_entry_point("router")
        
        # Add conditional edge from router
        # Routes to either "entity_extractor" (for data queries) or "generator" (for metadata)
        graph.add_conditional_edges(
            "router",
            should_route_to_analysis,
            {
                "entity_extractor": "entity_extractor",
                "generator": "generator",
            }
        )
        
        # Add sequential edges for the analysis path
        graph.add_edge("entity_extractor", "nl2sql")
        graph.add_edge("nl2sql", "executor")
        
        # Add conditional edge from executor (for reflection loop)
        graph.add_conditional_edges(
            "executor",
            should_reflect,
            {
                "reflection": "reflection",
                "generator": "generator",
            }
        )
        
        # Reflection loops back to nl2sql for retry
        graph.add_edge("reflection", "nl2sql")
        
        # Generator is the final node
        graph.add_edge("generator", END)
        
        return graph
    
    def invoke(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query through the graph.
        
        Args:
            query: User's natural language query
            thread_id: Optional thread ID for conversation continuity
            **kwargs: Additional configuration
            
        Returns:
            Final state containing response and metadata
        """
        # Create initial state
        initial_state = create_initial_state(query)
        
        # Configure execution
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        logger.info(f"[Graph] Processing query: {query[:100]}...")
        
        # Execute graph
        try:
            final_state = self.app.invoke(initial_state, config=config)
            
            # Add end time
            final_state["end_time"] = datetime.now().isoformat()
            
            logger.info(f"[Graph] Query processed successfully")
            
            return final_state
            
        except Exception as e:
            logger.error(f"[Graph] Execution failed: {e}")
            raise
    
    async def ainvoke(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async version of invoke.
        
        Args:
            query: User's natural language query
            thread_id: Optional thread ID
            **kwargs: Additional configuration
            
        Returns:
            Final state containing response
        """
        initial_state = create_initial_state(query)
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        logger.info(f"[Graph] Async processing query: {query[:100]}...")
        
        try:
            final_state = await self.app.ainvoke(initial_state, config=config)
            final_state["end_time"] = datetime.now().isoformat()
            return final_state
        except Exception as e:
            logger.error(f"[Graph] Async execution failed: {e}")
            raise
    
    def stream(
        self,
        query: str,
        thread_id: Optional[str] = None,
        **kwargs
    ):
        """
        Stream graph execution for real-time updates.
        
        Args:
            query: User's natural language query
            thread_id: Optional thread ID
            **kwargs: Additional configuration
            
        Yields:
            State updates from each node
        """
        initial_state = create_initial_state(query)
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        logger.info(f"[Graph] Streaming query: {query[:100]}...")
        
        for event in self.app.stream(initial_state, config=config):
            yield event
    
    def get_state(self, thread_id: str) -> Optional[AgentState]:
        """
        Get the current state for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Current state or None
        """
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)
    
    def get_graph_image(self) -> bytes:
        """
        Generate a visualization of the graph.
        
        Returns:
            PNG image bytes
        """
        return self.app.get_graph().draw_mermaid_png()


def create_graph(checkpointer: Optional[MemorySaver] = None) -> DataCopilotGraph:
    """
    Factory function to create a DataCopilotGraph instance.
    
    Args:
        checkpointer: Optional state checkpointer
        
    Returns:
        Configured DataCopilotGraph
    """
    return DataCopilotGraph(checkpointer=checkpointer)


# Convenience function for quick queries
def query(text: str) -> str:
    """
    Quick query function for simple use cases.
    
    Args:
        text: Natural language query
        
    Returns:
        Response string
    """
    graph = create_graph()
    result = graph.invoke(text)
    return result.get("final_response", "No response generated")
