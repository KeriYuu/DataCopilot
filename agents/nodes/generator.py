"""
Generator node for final response generation.

Uses Qwen2.5-32B to generate professional, actuary-style summaries
from SQL query results.
"""
from typing import Dict, Any, Optional
from loguru import logger

from data_copilot.agents.state import AgentState, IntentType, add_message
from data_copilot.models.qwen_adapter import get_generator
from data_copilot.utils.prompts import PromptTemplates, format_results_for_generator
from data_copilot.data.schema import SCHEMA_DESCRIPTION


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Final response generation node.
    
    Takes SQL query results and generates a professional,
    actuary-style summary using the Qwen2.5-32B model.
    
    For metadata queries, answers directly without SQL.
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with final response
    """
    user_query = state["user_query"]
    intent = state.get("intent", IntentType.UNKNOWN)
    sql_result = state.get("sql_result")
    generated_sql = state.get("generated_sql")
    execution_error = state.get("execution_error")
    
    logger.info(f"[Generator] Generating response for intent: {intent}")
    
    try:
        generator = get_generator()
        
        # Handle metadata questions (no SQL needed)
        if intent == IntentType.METADATA:
            response = _generate_metadata_response(generator, user_query)
        
        # Handle execution errors (max retries reached)
        elif execution_error and not sql_result:
            response = _generate_error_response(user_query, execution_error)
        
        # Handle successful SQL execution
        elif sql_result and sql_result.get("success"):
            response = _generate_data_response(
                generator, 
                user_query, 
                generated_sql, 
                sql_result
            )
        
        # Handle empty results
        elif sql_result and sql_result.get("row_count", 0) == 0:
            response = PromptTemplates.ERROR_NO_RESULTS
        
        # Fallback
        else:
            response = _generate_fallback_response(generator, user_query)
        
        logger.info(f"[Generator] Generated response ({len(response)} chars)")
        
        return {
            "final_response": response,
            "messages": [add_message(state, "assistant", response)],
        }
        
    except Exception as e:
        logger.error(f"[Generator] Generation failed: {e}")
        
        error_response = f"I encountered an issue generating a response: {str(e)}"
        
        return {
            "final_response": error_response,
            "messages": [add_message(state, "assistant", error_response)],
        }


def _generate_metadata_response(generator, query: str) -> str:
    """Generate response for metadata questions."""
    prompt = PromptTemplates.GENERATOR_METADATA.format(
        schema=SCHEMA_DESCRIPTION,
        query=query
    )
    
    messages = [
        {"role": "system", "content": PromptTemplates.GENERATOR_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    return generator.chat(messages, temperature=0.3)


def _generate_data_response(
    generator, 
    query: str, 
    sql: str, 
    result: dict
) -> str:
    """Generate response for successful data queries."""
    data = result.get("data", [])
    row_count = result.get("row_count", 0)
    
    formatted_results = format_results_for_generator(data)
    
    prompt = PromptTemplates.GENERATOR_USER.format(
        query=query,
        sql=sql,
        row_count=row_count,
        results=formatted_results
    )
    
    messages = [
        {"role": "system", "content": PromptTemplates.GENERATOR_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    return generator.chat(messages, temperature=0.3)


def _generate_error_response(query: str, error: str) -> str:
    """Generate response for queries that failed after max retries."""
    return (
        f"I was unable to complete your query: \"{query}\"\n\n"
        f"The issue encountered: {error}\n\n"
        f"{PromptTemplates.ERROR_MAX_RETRIES}"
    )


def _generate_fallback_response(generator, query: str) -> str:
    """Generate a fallback response when other methods fail."""
    messages = [
        {"role": "system", "content": PromptTemplates.GENERATOR_SYSTEM},
        {"role": "user", "content": f"The user asked: \"{query}\"\n\nI was unable to execute a database query for this request. Please provide a helpful response explaining what information might be available and how the user could rephrase their question."}
    ]
    
    return generator.chat(messages, temperature=0.3)
