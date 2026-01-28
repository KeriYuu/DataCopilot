"""
Evaluation metrics for Data Copilot.

Similar to test_score.py in the original system, this module
calculates accuracy and quality metrics for generated answers.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from data_copilot.evaluation.test_types import TestCase, TestResult, TestType


@dataclass
class MetricsReport:
    """Aggregated metrics report."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    # Score by type
    type_scores: Dict[str, float] = field(default_factory=dict)
    type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Detailed scores
    avg_value_score: float = 0.0
    avg_keyword_score: float = 0.0
    avg_semantic_score: float = 0.0
    avg_total_score: float = 0.0
    
    # Performance
    avg_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    
    # Errors
    error_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def weighted_score(self) -> float:
        """
        Calculate weighted final score similar to original test_score.py.
        
        Weights:
        - Type 1 (direct queries): 30%
        - Type 2 (formulas): 40%
        - Type 3-1 (open with data): 20%
        - Type 3-2 (general): 10%
        """
        weights = {
            "1": 0.15,
            "1-2": 0.15,
            "2-1": 0.25,
            "2-2": 0.15,
            "3-1": 0.20,
            "3-2": 0.10,
        }
        
        total = 0.0
        for type_key, weight in weights.items():
            if type_key in self.type_scores:
                total += self.type_scores[type_key] * weight
        
        return total * 100  # Convert to percentage
    
    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "accuracy": self.accuracy(),
                "weighted_score": self.weighted_score(),
            },
            "scores_by_type": self.type_scores,
            "counts_by_type": self.type_counts,
            "average_scores": {
                "value": self.avg_value_score,
                "keyword": self.avg_keyword_score,
                "semantic": self.avg_semantic_score,
                "total": self.avg_total_score,
            },
            "performance": {
                "avg_time_ms": self.avg_execution_time_ms,
                "total_time_ms": self.total_execution_time_ms,
            },
            "errors": {
                "count": self.error_count,
                "messages": self.error_messages[:10],  # Limit to 10
            }
        }


def calculate_value_score(
    expected: any,
    actual: str,
    tolerance: float = 0.01
) -> float:
    """
    Calculate score based on value correctness.
    
    Args:
        expected: Expected value(s)
        actual: Actual response
        tolerance: Tolerance for numeric comparisons
        
    Returns:
        Score between 0 and 1
    """
    if expected is None:
        return 0.5  # Neutral score if no expected value
    
    # Handle list of expected values
    expected_values = expected if isinstance(expected, list) else [expected]
    
    actual_clean = actual.replace(",", "").replace(" ", "").lower()
    
    for exp in expected_values:
        if exp is None:
            continue
            
        # Numeric comparison
        if isinstance(exp, (int, float)):
            numbers = re.findall(r'-?\d+\.?\d*', actual_clean)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if abs(num - exp) <= abs(exp * tolerance):
                        return 1.0
                except:
                    continue
        
        # String comparison
        else:
            exp_clean = str(exp).replace(",", "").replace(" ", "").lower()
            if exp_clean in actual_clean:
                return 1.0
    
    return 0.0


def calculate_keyword_score(
    keywords: List[str],
    actual: str
) -> float:
    """
    Calculate score based on keyword presence.
    
    Args:
        keywords: Required keywords
        actual: Actual response
        
    Returns:
        Score between 0 and 1
    """
    if not keywords:
        return 1.0  # No keywords required
    
    actual_lower = actual.lower()
    
    found = 0
    for keyword in keywords:
        if keyword.lower() in actual_lower:
            found += 1
    
    return found / len(keywords)


def calculate_semantic_score(
    expected_text: str,
    actual_text: str,
    model=None
) -> float:
    """
    Calculate semantic similarity score.
    
    Args:
        expected_text: Expected response text
        actual_text: Actual response text
        model: Optional sentence transformer model
        
    Returns:
        Score between 0 and 1
    """
    if not expected_text or not actual_text:
        return 0.5
    
    try:
        if model is None:
            # Try to load text2vec model (like original test_score.py)
            from text2vec import SentenceModel, semantic_search
            model = SentenceModel("shibing624/text2vec-base-chinese")
        
        # Calculate similarity
        embeddings = model.encode([expected_text, actual_text])
        similarity = semantic_search(
            embeddings[:1], embeddings[1:], top_k=1
        )[0][0]['score']
        
        return similarity
        
    except ImportError:
        # Fallback to simple word overlap
        expected_words = set(expected_text.lower().split())
        actual_words = set(actual_text.lower().split())
        
        if not expected_words:
            return 0.5
        
        overlap = len(expected_words & actual_words)
        return overlap / len(expected_words)
    
    except Exception:
        return 0.5


def calculate_sql_pattern_score(
    expected_pattern: Optional[str],
    actual_sql: Optional[str]
) -> float:
    """
    Check if generated SQL matches expected pattern.
    
    Args:
        expected_pattern: Regex pattern for expected SQL
        actual_sql: Generated SQL
        
    Returns:
        1.0 if matches, 0.0 otherwise
    """
    if not expected_pattern or not actual_sql:
        return 0.5
    
    try:
        if re.search(expected_pattern, actual_sql, re.IGNORECASE):
            return 1.0
    except:
        pass
    
    return 0.0


def score_result(
    result: TestResult,
    semantic_model=None
) -> TestResult:
    """
    Calculate all scores for a test result.
    
    Args:
        result: TestResult to score
        semantic_model: Optional model for semantic similarity
        
    Returns:
        Updated TestResult with scores
    """
    expected = result.test_case.expected
    
    # Value score
    result.value_score = calculate_value_score(
        expected.value,
        result.actual_answer,
        expected.tolerance
    )
    
    # Keyword score
    result.keyword_score = calculate_keyword_score(
        expected.keywords,
        result.actual_answer
    )
    
    # SQL pattern score (for formula tests)
    sql_score = calculate_sql_pattern_score(
        expected.sql_pattern,
        result.generated_sql
    )
    
    # Semantic score (optional)
    if isinstance(expected.value, str):
        result.semantic_score = calculate_semantic_score(
            str(expected.value),
            result.actual_answer,
            semantic_model
        )
    else:
        result.semantic_score = 0.5
    
    # Calculate total score based on test type
    test_type = result.test_case.test_type
    
    if test_type in [TestType.TYPE_1, TestType.TYPE_1_2]:
        # Direct queries: value + keyword
        result.total_score = (
            result.value_score * 0.5 +
            result.keyword_score * 0.5
        )
    elif test_type in [TestType.TYPE_2_1, TestType.TYPE_2_2]:
        # Formula queries: value + SQL pattern + keyword
        result.total_score = (
            result.value_score * 0.4 +
            sql_score * 0.3 +
            result.keyword_score * 0.3
        )
    else:
        # Open-ended: semantic + keyword
        result.total_score = (
            result.semantic_score * 0.6 +
            result.keyword_score * 0.4
        )
    
    # Determine pass/fail
    result.is_correct = result.total_score >= 0.5
    
    return result


def calculate_metrics(results: List[TestResult]) -> MetricsReport:
    """
    Calculate aggregated metrics from test results.
    
    Args:
        results: List of TestResult objects
        
    Returns:
        MetricsReport with aggregated metrics
    """
    report = MetricsReport()
    report.total_tests = len(results)
    
    # Group by type
    type_results = defaultdict(list)
    
    for result in results:
        type_key = result.test_case.test_type.value
        type_results[type_key].append(result)
        
        if result.is_correct:
            report.passed_tests += 1
        else:
            report.failed_tests += 1
        
        if result.error:
            report.error_count += 1
            report.error_messages.append(f"{result.test_case.id}: {result.error}")
        
        report.total_execution_time_ms += result.execution_time_ms
    
    # Calculate averages
    if results:
        report.avg_value_score = sum(r.value_score for r in results) / len(results)
        report.avg_keyword_score = sum(r.keyword_score for r in results) / len(results)
        report.avg_semantic_score = sum(r.semantic_score for r in results) / len(results)
        report.avg_total_score = sum(r.total_score for r in results) / len(results)
        report.avg_execution_time_ms = report.total_execution_time_ms / len(results)
    
    # Calculate scores by type
    for type_key, type_list in type_results.items():
        if type_list:
            report.type_scores[type_key] = sum(r.total_score for r in type_list) / len(type_list)
            report.type_counts[type_key] = len(type_list)
    
    return report


def print_report(report: MetricsReport):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("DATA COPILOT EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall Results:")
    print(f"  Total Tests: {report.total_tests}")
    print(f"  Passed: {report.passed_tests}")
    print(f"  Failed: {report.failed_tests}")
    print(f"  Accuracy: {report.accuracy():.2%}")
    print(f"  Weighted Score: {report.weighted_score():.2f}")
    
    print(f"\nScores by Type:")
    for type_key in sorted(report.type_scores.keys()):
        count = report.type_counts.get(type_key, 0)
        score = report.type_scores.get(type_key, 0)
        print(f"  Type {type_key}: {score:.2%} ({count} tests)")
    
    print(f"\nAverage Scores:")
    print(f"  Value Score: {report.avg_value_score:.2%}")
    print(f"  Keyword Score: {report.avg_keyword_score:.2%}")
    print(f"  Semantic Score: {report.avg_semantic_score:.2%}")
    
    print(f"\nPerformance:")
    print(f"  Average Time: {report.avg_execution_time_ms:.2f}ms")
    print(f"  Total Time: {report.total_execution_time_ms:.2f}ms")
    
    if report.error_count > 0:
        print(f"\nErrors: {report.error_count}")
        for msg in report.error_messages[:5]:
            print(f"  - {msg}")
    
    print("\n" + "=" * 60)
