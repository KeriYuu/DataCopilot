"""
SQL Executor node.

Executes generated SQL against the ClickHouse database.
"""
from typing import Dict, Any
from loguru import logger

from data_copilot.agents.state import AgentState, add_message
from data_copilot.tools.sql_executor import get_executor


def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    SQL execution node.
    
    Executes the generated SQL query against the database
    and captures results or errors.
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with execution results
    """
    sql = state.get("generated_sql")
    
    if not sql:
        logger.warning("[Executor] No SQL to execute")
        return {
            "sql_result": None,
            "execution_error": "No SQL query was generated",
            "messages": [add_message(state, "system", "No SQL to execute")],
        }
    
    logger.info(f"[Executor] Executing SQL: {sql[:100]}...")
    
    try:
        executor = get_executor()
        result = executor.execute(sql)
        
        if result.success:
            logger.info(f"[Executor] Query returned {result.row_count} rows in {result.execution_time_ms:.2f}ms")
            
            return {
                "sql_result": result.to_dict(),
                "execution_error": None,
                "messages": [add_message(state, "system", f"Query executed successfully: {result.row_count} rows in {result.execution_time_ms:.2f}ms")],
            }
        else:
            logger.warning(f"[Executor] Query failed: {result.error}")
            
            return {
                "sql_result": result.to_dict(),
                "execution_error": result.error,
                "messages": [add_message(state, "system", f"Query execution failed: {result.error}")],
            }
            
    except Exception as e:
        logger.error(f"[Executor] Execution error: {e}")
        
        return {
            "sql_result": None,
            "execution_error": str(e),
            "messages": [add_message(state, "system", f"Execution error: {e}")],
        }


def should_reflect(state: AgentState) -> str:
    """
    Conditional edge function for deciding whether to reflect/retry.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name ("reflection" or "generator")
    """
    from data_copilot.config import settings
    
    execution_error = state.get("execution_error")
    retry_count = state.get("retry_count", 0)
    sql_result = state.get("sql_result")
    
    # No error - proceed to generator
    if not execution_error:
        return "generator"
    
    # Max retries reached - give up and generate error response
    if retry_count >= settings.max_sql_retries:
        logger.warning(f"[Executor] Max retries ({settings.max_sql_retries}) reached")
        return "generator"
    
    # Has error but can retry - go to reflection
    logger.info(f"[Executor] Routing to reflection (attempt {retry_count + 1})")
    return "reflection"
