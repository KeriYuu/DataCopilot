"""
Main evaluator for Data Copilot.

This module orchestrates the evaluation process, similar to
generate_answer_with_classify.py in the original system.
"""
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from data_copilot.evaluation.test_types import (
    TestCase, TestResult, TestType,
    ALL_TEST_CASES, get_tests_by_type
)
from data_copilot.evaluation.metrics import (
    MetricsReport, calculate_metrics, score_result, print_report
)


class DataCopilotEvaluator:
    """
    Evaluator for the Data Copilot system.
    
    Runs test cases against the system and calculates metrics.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        use_tracing: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            api_url: URL of the Data Copilot API
            use_tracing: Whether to enable MLFlow tracing
        """
        self.api_url = api_url
        self.use_tracing = use_tracing
        self._graph = None
        self._traced_copilot = None
    
    def _get_graph(self):
        """Get or create the graph instance."""
        if self._graph is None:
            from data_copilot.agents.graph import create_graph
            self._graph = create_graph()
        return self._graph
    
    def _get_traced_copilot(self):
        """Get or create the traced copilot instance."""
        if self._traced_copilot is None and self.use_tracing:
            from data_copilot.observability import TracedDataCopilot
            self._traced_copilot = TracedDataCopilot()
        return self._traced_copilot
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: TestCase to run
            
        Returns:
            TestResult with scores
        """
        logger.info(f"Running test {test_case.id}: {test_case.question[:50]}...")
        
        start_time = time.time()
        
        try:
            # Run through the system
            if self.use_tracing and self._get_traced_copilot():
                state = self._get_traced_copilot().invoke(
                    query=test_case.question,
                    run_name=f"eval_{test_case.id}"
                )
            else:
                state = self._get_graph().invoke(test_case.question)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Extract results
            result = TestResult(
                test_case=test_case,
                actual_answer=state.get("final_response", ""),
                generated_sql=state.get("generated_sql"),
                sql_result=state.get("sql_result"),
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            logger.error(f"Test {test_case.id} failed: {e}")
            
            result = TestResult(
                test_case=test_case,
                actual_answer="",
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Score the result
        result = score_result(result)
        
        logger.info(f"Test {test_case.id}: score={result.total_score:.2f}, correct={result.is_correct}")
        
        return result
    
    def run_tests(
        self,
        test_cases: Optional[List[TestCase]] = None,
        test_type: Optional[TestType] = None,
        limit: Optional[int] = None
    ) -> List[TestResult]:
        """
        Run multiple test cases.
        
        Args:
            test_cases: Specific test cases to run (default: all)
            test_type: Filter by test type
            limit: Maximum number of tests to run
            
        Returns:
            List of TestResults
        """
        # Select tests
        if test_cases is not None:
            cases = test_cases
        elif test_type is not None:
            cases = get_tests_by_type(test_type)
        else:
            cases = ALL_TEST_CASES
        
        if limit:
            cases = cases[:limit]
        
        logger.info(f"Running {len(cases)} tests...")
        
        results = []
        for case in cases:
            result = self.run_single_test(case)
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        test_cases: Optional[List[TestCase]] = None,
        test_type: Optional[TestType] = None,
        output_path: Optional[str] = None,
        print_results: bool = True
    ) -> MetricsReport:
        """
        Run full evaluation and generate report.
        
        Args:
            test_cases: Specific tests to run
            test_type: Filter by type
            output_path: Path to save results
            print_results: Whether to print report
            
        Returns:
            MetricsReport with aggregated metrics
        """
        # Run tests
        results = self.run_tests(test_cases, test_type)
        
        # Calculate metrics
        report = calculate_metrics(results)
        
        # Save results
        if output_path:
            self._save_results(results, report, output_path)
        
        # Print report
        if print_results:
            print_report(report)
        
        return report
    
    def _save_results(
        self,
        results: List[TestResult],
        report: MetricsReport,
        output_path: str
    ):
        """Save results to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # Save report
        report_file = output_dir / f"report_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def run_formula_tests(self) -> MetricsReport:
        """Run formula calculation tests (Type 2-1, 2-2)."""
        from data_copilot.evaluation.test_types import get_formula_tests
        return self.evaluate(test_cases=get_formula_tests())
    
    def run_direct_query_tests(self) -> MetricsReport:
        """Run direct query tests (Type 1, 1-2)."""
        from data_copilot.evaluation.test_types import get_direct_query_tests
        return self.evaluate(test_cases=get_direct_query_tests())


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Data Copilot")
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "1", "1-2", "2-1", "2-2", "3-1", "3-2", "formula", "direct"],
        default="all",
        help="Test type to run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tests"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable MLFlow tracing"
    )
    
    args = parser.parse_args()
    
    evaluator = DataCopilotEvaluator(use_tracing=not args.no_trace)
    
    if args.type == "all":
        report = evaluator.evaluate(output_path=args.output)
    elif args.type == "formula":
        report = evaluator.run_formula_tests()
    elif args.type == "direct":
        report = evaluator.run_direct_query_tests()
    else:
        test_type = TestType(args.type)
        report = evaluator.evaluate(test_type=test_type, output_path=args.output)
    
    # Exit with appropriate code
    exit(0 if report.accuracy() >= 0.8 else 1)


if __name__ == "__main__":
    main()
