"""
Evaluation module for Data Copilot.

This module provides comprehensive testing and evaluation:
- Type 1: Direct query and aggregation tests (like original type1.py)
- Type 2: Formula calculation tests (like original type2.py)
- Type 3: Open-ended question tests
- E2E: End-to-end system tests
"""

from .test_types import TestType, TestCase, TestResult
from .evaluator import DataCopilotEvaluator
from .metrics import calculate_metrics, MetricsReport

__all__ = [
    "TestType",
    "TestCase", 
    "TestResult",
    "DataCopilotEvaluator",
    "calculate_metrics",
    "MetricsReport",
]
