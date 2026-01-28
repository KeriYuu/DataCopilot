"""
Testing module for evaluating trained LoRA adapters.

This module provides testing utilities for the three LoRA adapters:
- Intent LoRA: Intent classification (DATA_QUERY vs METADATA)
- Keyword LoRA: Entity extraction and normalization
- NL2SQL LoRA: Natural language to SQL conversion

Usage:
    # Run all LoRA tests
    python -m data_copilot.testing.run_tests --all
    
    # Run specific adapter test
    python -m data_copilot.testing.run_tests --adapter intent
    python -m data_copilot.testing.run_tests --adapter keyword
    python -m data_copilot.testing.run_tests --adapter nl2sql
"""

from data_copilot.testing.intent_tester import IntentTester
from data_copilot.testing.keyword_tester import KeywordTester
from data_copilot.testing.nl2sql_tester import NL2SQLTester

__all__ = [
    "IntentTester",
    "KeywordTester",
    "NL2SQLTester",
]
