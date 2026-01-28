"""
Reflection node for self-correction.

When SQL execution fails, this node analyzes the error and
prepares context for a retry attempt.
"""
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from data_copilot.agents.state import AgentState, add_message


def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Reflection/error handling node.
    
    Analyzes SQL execution errors and prepares context for
    the NL2SQL generator to produce a corrected query.
    
    This implements the "Reflection" loop where the system
    injects specific error messages into the prompt history,
    allowing the model to "reflect" on its mistake.
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with reflection context
    """
    execution_error = state.get("execution_error", "Unknown error")
    generated_sql = state.get("generated_sql", "")
    retry_count = state.get("retry_count", 0)
    reflection_steps = state.get("reflection_steps", [])
    
    logger.info(f"[Reflection] Analyzing error (attempt {retry_count + 1}): {execution_error[:100]}...")
    
    # Create reflection step record
    reflection_step = {
        "attempt": retry_count + 1,
        "original_sql": generated_sql,
        "error_message": execution_error,
        "corrected_sql": None,
        "timestamp": datetime.now().isoformat(),
        "analysis": _analyze_error(execution_error),
    }
    
    # Add to reflection history
    updated_reflection_steps = reflection_steps + [reflection_step]
    
    logger.info(f"[Reflection] Error analysis: {reflection_step['analysis']}")
    
    return {
        "reflection_steps": updated_reflection_steps,
        "retry_count": retry_count + 1,
        "messages": [add_message(state, "system", f"Reflection: {reflection_step['analysis']}")],
    }


def _analyze_error(error: str) -> str:
    """
    Analyze SQL error and provide guidance for correction.
    
    Args:
        error: Error message from database or validation
        
    Returns:
        Analysis and guidance for correction
    """
    error_lower = error.lower()
    
    # Column not found
    if "no such column" in error_lower or "unknown column" in error_lower or "column" in error_lower:
        return "Column name error - verify column names match the schema exactly"
    
    # Syntax error
    if "syntax" in error_lower:
        return "SQL syntax error - check query structure, quotes, and parentheses"
    
    # Group by error
    if "group by" in error_lower or "aggregate" in error_lower:
        return "Aggregation error - ensure all non-aggregated columns are in GROUP BY"
    
    # Type mismatch
    if "type" in error_lower or "cannot convert" in error_lower:
        return "Data type mismatch - verify value types match column types"
    
    # Division by zero
    if "division" in error_lower or "zero" in error_lower:
        return "Division by zero - use NULLIF to prevent division by zero"
    
    # Table not found
    if "table" in error_lower and ("not found" in error_lower or "doesn't exist" in error_lower):
        return "Table not found - verify table name is correct"
    
    # Permission error
    if "permission" in error_lower or "access denied" in error_lower:
        return "Permission error - this is a system issue, not query-related"
    
    # Timeout
    if "timeout" in error_lower:
        return "Query timeout - consider adding more filters or LIMIT clause"
    
    # Generic
    return f"Query error: {error[:200]}. Please review and correct the SQL."
