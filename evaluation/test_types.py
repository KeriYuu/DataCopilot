"""
Test type definitions for evaluation.

Similar to the original type1.py and type2.py, this module defines
different categories of test cases for the insurance data system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class TestType(str, Enum):
    """
    Test categories similar to original system.
    
    - TYPE_1: Direct queries (simple lookups, single aggregations)
    - TYPE_1_2: Multi-field queries (multiple values in one question)
    - TYPE_2_1: Formula calculations (loss ratio, frequency, etc.)
    - TYPE_2_2: Comparison queries (year-over-year, state vs state)
    - TYPE_3_1: Open-ended with data (summarize, analyze)
    - TYPE_3_2: General knowledge (definitions, explanations)
    """
    TYPE_1 = "1"          # Direct queries
    TYPE_1_2 = "1-2"      # Multi-field queries
    TYPE_2_1 = "2-1"      # Formula calculations  
    TYPE_2_2 = "2-2"      # Comparison queries
    TYPE_3_1 = "3-1"      # Open-ended with data
    TYPE_3_2 = "3-2"      # General knowledge


@dataclass
class ExpectedAnswer:
    """Expected answer for a test case."""
    value: Any                          # Expected value or list of values
    tolerance: float = 0.01             # Tolerance for numeric comparisons
    keywords: List[str] = field(default_factory=list)  # Required keywords in response
    sql_pattern: Optional[str] = None   # Expected SQL pattern (regex)


@dataclass
class TestPrompt:
    """Prompt configuration for a test case."""
    year: Optional[str] = None
    state: Optional[str] = None
    class_code: Optional[str] = None
    coverage_type: Optional[str] = None
    key_word: Optional[str] = None
    formula: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TestCase:
    """
    A single test case for evaluation.
    
    Similar to the test format in the original system.
    """
    id: str
    question: str
    test_type: TestType
    expected: ExpectedAnswer
    prompt: TestPrompt = field(default_factory=TestPrompt)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question": self.question,
            "type": self.test_type.value,
            "answer": self.expected.value if isinstance(self.expected.value, list) else [self.expected.value],
            "prompt": self.prompt.to_dict(),
        }


@dataclass
class TestResult:
    """Result of running a test case."""
    test_case: TestCase
    actual_answer: str
    generated_sql: Optional[str] = None
    sql_result: Optional[Dict] = None
    
    # Scoring
    is_correct: bool = False
    value_score: float = 0.0      # Score for value correctness
    keyword_score: float = 0.0     # Score for keyword presence
    semantic_score: float = 0.0    # Semantic similarity score
    total_score: float = 0.0
    
    # Metadata
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.test_case.id,
            "question": self.test_case.question,
            "type": self.test_case.test_type.value,
            "expected": self.test_case.expected.value,
            "actual": self.actual_answer,
            "sql": self.generated_sql,
            "is_correct": self.is_correct,
            "scores": {
                "value": self.value_score,
                "keyword": self.keyword_score,
                "semantic": self.semantic_score,
                "total": self.total_score,
            },
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


# ===================== TYPE 1: DIRECT QUERIES =====================

TYPE_1_TESTS: List[TestCase] = [
    TestCase(
        id="t1_001",
        question="What is the total written premium for CA in 2023?",
        test_type=TestType.TYPE_1,
        expected=ExpectedAnswer(
            value=None,  # Will be filled with actual data
            keywords=["written_premium", "CA", "2023"]
        ),
        prompt=TestPrompt(year="2023", state="CA", key_word="written_premium")
    ),
    TestCase(
        id="t1_002",
        question="Show total claim count by state",
        test_type=TestType.TYPE_1,
        expected=ExpectedAnswer(
            value=None,
            keywords=["claim_count", "state"]
        ),
        prompt=TestPrompt(key_word="claim_count")
    ),
    TestCase(
        id="t1_003",
        question="How many claims in Texas for Workers Compensation?",
        test_type=TestType.TYPE_1,
        expected=ExpectedAnswer(
            value=None,
            keywords=["claim", "TX", "Workers Compensation"]
        ),
        prompt=TestPrompt(state="TX", coverage_type="Workers Compensation", key_word="claim_count")
    ),
]


# ===================== TYPE 1-2: MULTI-FIELD QUERIES =====================

TYPE_1_2_TESTS: List[TestCase] = [
    TestCase(
        id="t1_2_001",
        question="What are the written premium and incurred loss for CA in 2023?",
        test_type=TestType.TYPE_1_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["written_premium", "incurred_loss", "CA", "2023"]
        ),
        prompt=TestPrompt(year="2023", state="CA", key_word="written_premium,incurred_loss")
    ),
    TestCase(
        id="t1_2_002",
        question="Show claim count and paid loss by coverage type",
        test_type=TestType.TYPE_1_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["claim_count", "paid_loss", "coverage_type"]
        ),
        prompt=TestPrompt(key_word="claim_count,paid_loss")
    ),
]


# ===================== TYPE 2-1: FORMULA CALCULATIONS =====================

TYPE_2_1_TESTS: List[TestCase] = [
    TestCase(
        id="t2_1_001",
        question="What is the loss ratio for California in 2023?",
        test_type=TestType.TYPE_2_1,
        expected=ExpectedAnswer(
            value=None,
            tolerance=0.01,
            keywords=["loss ratio", "CA", "2023"],
            sql_pattern=r"SUM\(incurred_loss\).*SUM\(earned_premium\)"
        ),
        prompt=TestPrompt(year="2023", state="CA", key_word="loss_ratio", formula="loss_ratio")
    ),
    TestCase(
        id="t2_1_002",
        question="Calculate the pure premium for class code 8810",
        test_type=TestType.TYPE_2_1,
        expected=ExpectedAnswer(
            value=None,
            tolerance=0.01,
            keywords=["pure premium", "8810"],
            sql_pattern=r"SUM\(incurred_loss\).*SUM\(exposure_units\)"
        ),
        prompt=TestPrompt(class_code="8810", key_word="pure_premium", formula="pure_premium")
    ),
    TestCase(
        id="t2_1_003",
        question="What is the claim frequency by state?",
        test_type=TestType.TYPE_2_1,
        expected=ExpectedAnswer(
            value=None,
            tolerance=0.01,
            keywords=["frequency", "state"],
            sql_pattern=r"SUM\(claim_count\).*SUM\(exposure_units\)"
        ),
        prompt=TestPrompt(key_word="frequency", formula="frequency")
    ),
    TestCase(
        id="t2_1_004",
        question="Calculate severity for Workers Compensation",
        test_type=TestType.TYPE_2_1,
        expected=ExpectedAnswer(
            value=None,
            tolerance=0.01,
            keywords=["severity", "Workers Compensation"],
            sql_pattern=r"SUM\(incurred_loss\).*SUM\(claim_count\)"
        ),
        prompt=TestPrompt(coverage_type="Workers Compensation", key_word="severity", formula="severity")
    ),
    TestCase(
        id="t2_1_005",
        question="Show combined ratio by region",
        test_type=TestType.TYPE_2_1,
        expected=ExpectedAnswer(
            value=None,
            tolerance=0.01,
            keywords=["combined ratio", "region"],
            sql_pattern=r"SUM\(incurred_loss\).*SUM\(expense\).*SUM\(earned_premium\)"
        ),
        prompt=TestPrompt(key_word="combined_ratio", formula="combined_ratio")
    ),
]


# ===================== TYPE 2-2: COMPARISON QUERIES =====================

TYPE_2_2_TESTS: List[TestCase] = [
    TestCase(
        id="t2_2_001",
        question="Compare loss ratio between CA and NY for 2023",
        test_type=TestType.TYPE_2_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["loss ratio", "CA", "NY", "2023"]
        ),
        prompt=TestPrompt(year="2023", key_word="loss_ratio")
    ),
    TestCase(
        id="t2_2_002",
        question="Is the 2023 loss ratio higher or lower than 2022 for Texas?",
        test_type=TestType.TYPE_2_2,
        expected=ExpectedAnswer(
            value=None,  # "higher" or "lower" or "same"
            keywords=["loss ratio", "2023", "2022", "TX"]
        ),
        prompt=TestPrompt(state="TX", year="2023", key_word="loss_ratio")
    ),
    TestCase(
        id="t2_2_003",
        question="Which state has higher severity: California or Texas?",
        test_type=TestType.TYPE_2_2,
        expected=ExpectedAnswer(
            value=None,  # "CA" or "TX"
            keywords=["severity", "California", "Texas"]
        ),
        prompt=TestPrompt(key_word="severity")
    ),
]


# ===================== TYPE 3-1: OPEN-ENDED WITH DATA =====================

TYPE_3_1_TESTS: List[TestCase] = [
    TestCase(
        id="t3_1_001",
        question="Summarize the loss performance for California Workers Compensation",
        test_type=TestType.TYPE_3_1,
        expected=ExpectedAnswer(
            value=None,
            keywords=["loss", "CA", "Workers Compensation", "premium"]
        ),
        prompt=TestPrompt(state="CA", coverage_type="Workers Compensation")
    ),
    TestCase(
        id="t3_1_002",
        question="Analyze the claim trends for class code 8810",
        test_type=TestType.TYPE_3_1,
        expected=ExpectedAnswer(
            value=None,
            keywords=["claim", "8810", "trend"]
        ),
        prompt=TestPrompt(class_code="8810", key_word="claim_count")
    ),
]


# ===================== TYPE 3-2: GENERAL KNOWLEDGE =====================

TYPE_3_2_TESTS: List[TestCase] = [
    TestCase(
        id="t3_2_001",
        question="What is loss ratio and how is it calculated?",
        test_type=TestType.TYPE_3_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["loss ratio", "incurred", "premium", "formula"]
        ),
        prompt=TestPrompt(key_word="loss_ratio")
    ),
    TestCase(
        id="t3_2_002",
        question="What is class code 8810?",
        test_type=TestType.TYPE_3_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["8810", "clerical", "office"]
        ),
        prompt=TestPrompt(class_code="8810")
    ),
    TestCase(
        id="t3_2_003",
        question="What columns are available in the insurance database?",
        test_type=TestType.TYPE_3_2,
        expected=ExpectedAnswer(
            value=None,
            keywords=["premium", "loss", "claim", "exposure", "state"]
        ),
        prompt=TestPrompt()
    ),
]


# All test cases combined
ALL_TEST_CASES: List[TestCase] = (
    TYPE_1_TESTS +
    TYPE_1_2_TESTS +
    TYPE_2_1_TESTS +
    TYPE_2_2_TESTS +
    TYPE_3_1_TESTS +
    TYPE_3_2_TESTS
)


def get_tests_by_type(test_type: TestType) -> List[TestCase]:
    """Get all test cases of a specific type."""
    return [t for t in ALL_TEST_CASES if t.test_type == test_type]


def get_formula_tests() -> List[TestCase]:
    """Get all formula calculation tests (similar to original type2)."""
    return TYPE_2_1_TESTS + TYPE_2_2_TESTS


def get_direct_query_tests() -> List[TestCase]:
    """Get all direct query tests (similar to original type1)."""
    return TYPE_1_TESTS + TYPE_1_2_TESTS
