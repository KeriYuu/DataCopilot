"""
Entity Extractor node for keyword/entity rewriting.

Resolves ambiguity before SQL generation by mapping colloquial
inputs to schema-valid values.
"""
import json
import re
from typing import Dict, Any, Optional
from loguru import logger

from data_copilot.agents.state import AgentState, add_message
from data_copilot.models.lora_router import get_router, LoRAType
from data_copilot.utils.prompts import PromptTemplates
from data_copilot.data.schema import SCHEMA_DESCRIPTION, ENTITY_MAPPINGS


def entity_extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Entity extraction and rewriting node.
    
    Maps colloquial inputs to schema-valid values:
    - "Workers Comp" -> class_group="WC"
    - "SoCal" -> state="CA" or region="West"
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with extracted entities and rewritten query
    """
    user_query = state["user_query"]
    
    logger.info(f"[EntityExtractor] Extracting entities from query...")
    
    # First try rule-based extraction for common patterns
    rule_based_entities = _extract_entities_rule_based(user_query)
    
    try:
        router = get_router()
        
        # Build the extraction prompt
        prompt = (
            f"{PromptTemplates.ENTITY_SYSTEM.format(schema=SCHEMA_DESCRIPTION)}\n\n"
            f"{PromptTemplates.ENTITY_USER.format(query=user_query)}"
        )
        
        # Get extraction from Keyword LoRA
        response = router.generate(
            prompt=prompt,
            lora_type=LoRAType.KEYWORD,
            max_tokens=500,
            temperature=0.0
        )
        
        # Parse JSON response
        model_entities, rewritten = _parse_entity_response(response)
        
        # Merge rule-based and model-based entities (rule-based takes precedence)
        entities = {**model_entities, **rule_based_entities}
        
        logger.info(f"[EntityExtractor] Extracted entities: {entities}")
        
        return {
            "extracted_entities": entities,
            "rewritten_query": rewritten or user_query,
            "messages": [add_message(state, "system", f"Extracted entities: {json.dumps(entities)}")],
        }
        
    except Exception as e:
        logger.warning(f"[EntityExtractor] Model extraction failed: {e}, using rule-based only")
        
        return {
            "extracted_entities": rule_based_entities,
            "rewritten_query": user_query,
            "messages": [add_message(state, "system", f"Extracted entities (rule-based): {json.dumps(rule_based_entities)}")],
        }


def _extract_entities_rule_based(query: str) -> Dict[str, str]:
    """
    Rule-based entity extraction using predefined mappings.
    
    Args:
        query: User query
        
    Returns:
        Dictionary of extracted entities
    """
    entities = {}
    query_lower = query.lower()
    
    for column, mappings in ENTITY_MAPPINGS.items():
        for phrase, value in mappings.items():
            if phrase in query_lower:
                entities[column] = value
                break  # Take first match for each column
    
    # Extract year patterns
    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        year = int(year_match.group(1))
        # Determine if it's policy_year or accident_year based on context
        if 'accident' in query_lower or 'loss' in query_lower:
            entities['accident_year'] = year
        elif 'policy' in query_lower:
            entities['policy_year'] = year
        else:
            # Default to policy_year
            entities['policy_year'] = year
    
    # Extract class codes (4-digit patterns that look like class codes)
    class_code_match = re.search(r'\b(8810|8742|5506|8820|8017|9015|5183|5190|\d{4})\b', query)
    if class_code_match and 'class' in query_lower:
        entities['class_code'] = class_code_match.group(1)
    
    # Extract state codes (2-letter uppercase)
    state_match = re.search(r'\b([A-Z]{2})\b', query)
    if state_match and state_match.group(1) in ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']:
        entities['state'] = state_match.group(1)
    
    # Extract ZIP codes
    zip_match = re.search(r'\b(\d{5})\b', query)
    if zip_match and 'zip' in query_lower:
        entities['zip_code'] = zip_match.group(1)
    
    return entities


def _parse_entity_response(response: str) -> tuple[Dict[str, str], Optional[str]]:
    """
    Parse the entity extraction response.
    
    Args:
        response: Model response (should be JSON)
        
    Returns:
        Tuple of (entities_dict, rewritten_query)
    """
    entities = {}
    rewritten = None
    
    # Try to extract JSON from response
    try:
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            entities = data.get("entities", {})
            rewritten = data.get("rewritten_query")
    except json.JSONDecodeError:
        logger.warning(f"[EntityExtractor] Failed to parse JSON: {response[:200]}")
    
    return entities, rewritten
