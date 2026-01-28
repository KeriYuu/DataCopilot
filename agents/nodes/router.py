"""
Router node for intent classification.

The Router analyzes user queries and directs them to the appropriate
downstream task based on intent classification.

Simplified to 2 intent types:
- DATA_QUERY: All data-related queries (lookups, aggregations, calculations)
- METADATA: Definitions, system info, and general questions
"""
import re
from typing import Dict, Any
from loguru import logger

from data_copilot.agents.state import AgentState, IntentType, add_message
from data_copilot.models.lora_router import get_router, LoRAType
from data_copilot.utils.prompts import PromptTemplates


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Intent classification node.
    
    Classifies the user query into one of two categories:
    - DATA_QUERY: All data-related queries requiring SQL
    - METADATA: Definitions and general information (no SQL needed)
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with intent
    """
    user_query = state["user_query"]
    
    logger.info(f"[Router] Classifying query: {user_query[:100]}...")
    
    try:
        router = get_router()
        
        # Build the classification prompt
        prompt = f"{PromptTemplates.INTENT_SYSTEM}\n\n{PromptTemplates.INTENT_USER.format(query=user_query)}"
        
        # Get classification from Intent LoRA
        response = router.generate(
            prompt=prompt,
            lora_type=LoRAType.INTENT,
            max_tokens=20,
            temperature=0.0
        )
        
        # Parse response (format: CATEGORY)
        intent = _parse_intent_response(response)
        
        logger.info(f"[Router] Classified as {intent.value}")
        
        return {
            "intent": intent,
            "messages": [add_message(state, "system", f"Intent: {intent.value}")],
        }
        
    except Exception as e:
        logger.error(f"[Router] Classification failed: {e}")
        # Default to DATA_QUERY on error (most common case)
        return {
            "intent": IntentType.DATA_QUERY,
            "messages": [add_message(state, "system", f"Intent classification failed, defaulting to DATA_QUERY")],
        }


def _parse_intent_response(response: str) -> IntentType:
    """
    Parse the intent classification response.
    
    Args:
        response: Model response (category name only)
        
    Returns:
        IntentType enum value
    """
    response = response.strip().upper()
    
    # Remove any potential confidence scores or extra text
    if "|" in response:
        response = response.split("|")[0].strip()
    
    # Take first word only
    response = response.split()[0] if response else "UNKNOWN"
    
    # Map to IntentType
    intent_map = {
        # DATA_QUERY mappings
        "DATA_QUERY": IntentType.DATA_QUERY,
        "DATA": IntentType.DATA_QUERY,
        "QUERY": IntentType.DATA_QUERY,
        # Legacy mappings (for backward compatibility during transition)
        "DIRECT_QUERY": IntentType.DATA_QUERY,
        "DIRECT": IntentType.DATA_QUERY,
        "STATISTICAL_AGGREGATION": IntentType.DATA_QUERY,
        "STATISTICAL": IntentType.DATA_QUERY,
        "AGGREGATION": IntentType.DATA_QUERY,
        # METADATA mappings
        "METADATA": IntentType.METADATA,
        "META": IntentType.METADATA,
        "GENERAL": IntentType.METADATA,
        # Legacy mapping
        "METADATA_GENERAL": IntentType.METADATA,
    }
    
    return intent_map.get(response, IntentType.UNKNOWN)


def should_route_to_analysis(state: AgentState) -> str:
    """
    Conditional edge function for routing decisions.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name based on intent:
        - "entity_extractor": for DATA_QUERY (needs SQL)
        - "generator": for METADATA (direct answer, no SQL)
    """
    intent = state.get("intent", IntentType.UNKNOWN)
    
    if intent == IntentType.METADATA:
        logger.info("[Router] Routing to generator (metadata query)")
        return "generator"
    else:
        # DATA_QUERY and UNKNOWN both go through analysis path
        logger.info("[Router] Routing to entity_extractor (data query)")
        return "entity_extractor"
