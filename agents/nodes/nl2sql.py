"""
NL2SQL Generator node.

Converts natural language queries to ClickHouse-compatible SQL.
"""
import json
from typing import Dict, Any
from loguru import logger

from data_copilot.agents.state import AgentState, add_message
from data_copilot.models.lora_router import get_router, LoRAType
from data_copilot.utils.prompts import PromptTemplates, format_entities_context
from data_copilot.utils.sql_utils import SQLValidator, extract_sql_from_response


def nl2sql_node(state: AgentState) -> Dict[str, Any]:
    """
    SQL generation node.
    
    Converts natural language to ClickHouse-compatible SQL using
    the NL2SQL LoRA adapter. Includes domain-specific knowledge
    about insurance formulas.
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with generated SQL
    """
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query") or user_query
    entities = state.get("extracted_entities", {})
    reflection_steps = state.get("reflection_steps", [])
    
    logger.info(f"[NL2SQL] Generating SQL for: {rewritten_query[:100]}...")
    
    try:
        router = get_router()
        validator = SQLValidator()
        
        # Check if this is a retry (has reflection context)
        if reflection_steps:
            last_reflection = reflection_steps[-1]
            prompt = (
                f"{PromptTemplates.NL2SQL_SYSTEM}\n\n"
                f"{PromptTemplates.NL2SQL_REFLECTION.format(previous_sql=last_reflection['original_sql'], error_message=last_reflection['error_message'])}"
            )
        else:
            # Build the generation prompt
            entities_context = format_entities_context(entities)
            prompt = (
                f"{PromptTemplates.NL2SQL_SYSTEM}\n\n"
                f"{PromptTemplates.NL2SQL_USER.format(query=rewritten_query, entities_context=entities_context)}"
            )
        
        # Generate SQL using NL2SQL LoRA
        response = router.generate(
            prompt=prompt,
            lora_type=LoRAType.NL2SQL,
            max_tokens=1024,
            temperature=0.0
        )
        
        # Extract and validate SQL
        sql = extract_sql_from_response(response)
        sql = validator.sanitize(sql)
        
        is_valid, error = validator.validate(sql)
        if not is_valid:
            logger.warning(f"[NL2SQL] Generated SQL failed validation: {error}")
            # Store the error for potential reflection
            return {
                "generated_sql": sql,
                "execution_error": error,
                "messages": [add_message(state, "system", f"Generated SQL (validation warning): {sql[:200]}...")],
            }
        
        logger.info(f"[NL2SQL] Generated SQL: {sql[:200]}...")
        
        return {
            "generated_sql": sql,
            "sql_explanation": _generate_sql_explanation(sql),
            "execution_error": None,
            "messages": [add_message(state, "system", f"Generated SQL: {sql}")],
        }
        
    except Exception as e:
        logger.error(f"[NL2SQL] Generation failed: {e}")
        return {
            "generated_sql": None,
            "execution_error": str(e),
            "messages": [add_message(state, "system", f"SQL generation failed: {e}")],
        }


def _generate_sql_explanation(sql: str) -> str:
    """
    Generate a brief explanation of what the SQL does.
    
    Args:
        sql: Generated SQL query
        
    Returns:
        Human-readable explanation
    """
    explanation_parts = []
    sql_upper = sql.upper()
    
    # Check for aggregations
    if 'SUM(' in sql_upper:
        explanation_parts.append("aggregates data using SUM")
    if 'AVG(' in sql_upper:
        explanation_parts.append("calculates averages")
    if 'COUNT(' in sql_upper:
        explanation_parts.append("counts records")
    
    # Check for grouping
    if 'GROUP BY' in sql_upper:
        explanation_parts.append("groups results")
    
    # Check for filtering
    if 'WHERE' in sql_upper:
        explanation_parts.append("filters data based on conditions")
    
    # Check for ordering
    if 'ORDER BY' in sql_upper:
        explanation_parts.append("sorts results")
    
    # Check for limiting
    if 'LIMIT' in sql_upper:
        explanation_parts.append("limits output rows")
    
    if explanation_parts:
        return f"This query {', '.join(explanation_parts)}."
    else:
        return "This query retrieves data from the database."
