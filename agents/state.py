"""
Agent state definition for LangGraph.

The AgentState is a shared object that persists across nodes,
maintaining the chat history, generated SQL, execution results, 
and any error logs.
"""
from typing import TypedDict, List, Optional, Any, Annotated
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import operator


class IntentType(str, Enum):
    """
    Intent classification categories (simplified to 2 types).
    
    - DATA_QUERY: All data-related queries (lookups, aggregations, calculations)
      Examples: "Show me policy X", "Calculate loss ratio by state"
    - METADATA: Definitions, system info, and general questions
      Examples: "What is class code 8810?", "What columns are available?"
    """
    DATA_QUERY = "data_query"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SQLResult:
    """Result of SQL execution."""
    sql: str
    success: bool
    data: Optional[List[dict]] = None
    error: Optional[str] = None
    row_count: int = 0
    execution_time_ms: float = 0.0


@dataclass
class ReflectionStep:
    """A single reflection/retry step."""
    attempt: int
    original_sql: str
    error_message: str
    corrected_sql: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AgentState(TypedDict):
    """
    Shared state object for the LangGraph agent.
    
    This state persists across all nodes in the graph, allowing
    downstream nodes to access the full context of the "thought process".
    """
    # Input
    user_query: str
    
    # Chat history (accumulated with operator.add)
    messages: Annotated[List[dict], operator.add]
    
    # Intent classification (simplified - no confidence)
    intent: Optional[IntentType]
    
    # Entity extraction
    extracted_entities: dict  # {column_name: extracted_value}
    rewritten_query: Optional[str]
    
    # SQL generation
    generated_sql: Optional[str]
    sql_explanation: Optional[str]
    
    # Execution
    sql_result: Optional[dict]  # Serialized SQLResult
    execution_error: Optional[str]
    
    # Reflection/retry
    reflection_steps: List[dict]  # Serialized ReflectionStep list
    retry_count: int
    
    # Final output
    final_response: Optional[str]
    
    # Metadata
    start_time: Optional[str]
    end_time: Optional[str]
    total_tokens: int


def create_initial_state(user_query: str) -> AgentState:
    """Create an initial state for a new query."""
    return AgentState(
        user_query=user_query,
        messages=[{"role": "user", "content": user_query}],
        intent=None,
        extracted_entities={},
        rewritten_query=None,
        generated_sql=None,
        sql_explanation=None,
        sql_result=None,
        execution_error=None,
        reflection_steps=[],
        retry_count=0,
        final_response=None,
        start_time=datetime.now().isoformat(),
        end_time=None,
        total_tokens=0,
    )


def add_message(state: AgentState, role: str, content: str) -> dict:
    """Helper to create a message dict for state update."""
    return {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
