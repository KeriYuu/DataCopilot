"""
NL2SQL LoRA Tester.

Tests the NL2SQL LoRA adapter which converts natural language queries
to ClickHouse-compatible SQL for the insurance dataset.
"""
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from data_copilot.tools.sql_executor import SQLExecutor

@dataclass
class NL2SQLTestResult:
    """Result of a single NL2SQL test."""
    query: str
    expected_sql: str
    predicted_sql: str
    sql_match: bool
    sql_similarity: float
    execution_success: bool
    execution_error: Optional[str]
    numeric_match: bool
    numeric_similarity: float
    inference_time_ms: float
    raw_output: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "expected_sql": self.expected_sql,
            "predicted_sql": self.predicted_sql,
            "sql_match": self.sql_match,
            "sql_similarity": self.sql_similarity,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "numeric_match": self.numeric_match,
            "numeric_similarity": self.numeric_similarity,
            "time_ms": self.inference_time_ms,
            "raw_output": self.raw_output,
        }


@dataclass
class NL2SQLTestReport:
    """Aggregated test report for NL2SQL LoRA."""
    total_tests: int = 0
    correct_predictions: int = 0
    
    # Similarity metrics
    avg_sql_similarity: float = 0.0
    avg_numeric_similarity: float = 0.0
    numeric_evaluated: int = 0
    numeric_correct: int = 0
    execution_successes: int = 0
    
    # Performance
    avg_inference_time_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    
    # Results
    results: List[NL2SQLTestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy based on SQL match."""
        if self.total_tests == 0:
            return 0.0
        return self.correct_predictions / self.total_tests
    
    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_tests": self.total_tests,
                "correct": self.correct_predictions,
                "accuracy": self.accuracy,
            },
            "sql_similarity": {
                "average": self.avg_sql_similarity,
            },
            "execution": {
                "successes": self.execution_successes,
                "success_rate": self.execution_successes / self.total_tests if self.total_tests else 0.0,
            },
            "numeric_accuracy": {
                "evaluated": self.numeric_evaluated,
                "correct": self.numeric_correct,
                "accuracy": self.numeric_correct / self.numeric_evaluated if self.numeric_evaluated else 0.0,
                "average_similarity": self.avg_numeric_similarity,
            },
            "performance": {
                "avg_time_ms": self.avg_inference_time_ms,
                "total_time_ms": self.total_inference_time_ms,
            },
            "errors": self.errors[:10],
        }


class NL2SQLTester:
    """
    Tester for NL2SQL LoRA.
    
    Tests the model's ability to generate correct SQL queries from
    natural language prompts.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        dataframe_path: Optional[str] = None,
        device: str = "cuda",
        numeric_tolerance: float = 1e-3,
    ):
        """
        Initialize the NL2SQL tester.
        
        Args:
            model_path: Path to merged model (if using merged model)
            base_model: Base model name/path
            adapter_path: Path to LoRA adapter (if not merged)
            device: Device to run inference on
        """
        self.model_path = model_path
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.dataframe_path = dataframe_path
        self.device = device
        self.numeric_tolerance = numeric_tolerance
        
        self._model = None
        self._tokenizer = None
        self._executor = None
    
    def _load_model(self):
        """Load model and tokenizer (lazy loading)."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info("Loading NL2SQL LoRA model...")
            
            # Load tokenizer
            model_name = self.model_path or self.base_model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            if self.model_path:
                # Use merged model
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True
                )
            elif self.adapter_path:
                # Load base + adapter
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True
                )
                self._model = PeftModel.from_pretrained(base, self.adapter_path)
            else:
                raise ValueError("Must provide either model_path or adapter_path")
            
            self._model.eval()
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.info("Install with: pip install transformers peft torch")
            raise

    def _load_executor(self):
        """Load SQL executor with optional DataFrame."""
        if self._executor is not None:
            return
        
        if self.dataframe_path:
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError("pandas required for DataFrame execution. Install with: pip install pandas") from e
            
            df = pd.read_csv(self.dataframe_path)
            self._executor = SQLExecutor(use_dataframe=True, dataframe=df)
        else:
            # Fallback to default executor (ClickHouse or configured)
            self._executor = SQLExecutor()
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        if not sql:
            return ""
        
        # Remove extra whitespace and normalize case
        normalized = re.sub(r"\s+", " ", sql.strip())
        normalized = normalized.replace("`", "")
        return normalized.lower()
    
    def _sql_similarity(self, expected: str, predicted: str) -> float:
        """Calculate SQL similarity based on token overlap."""
        if not expected or not predicted:
            return 0.0
        
        expected_norm = self._normalize_sql(expected)
        predicted_norm = self._normalize_sql(predicted)
        
        # Token-based overlap
        expected_tokens = set(expected_norm.split())
        predicted_tokens = set(predicted_norm.split())
        
        if not expected_tokens:
            return 0.0
        
        overlap = len(expected_tokens & predicted_tokens)
        return overlap / len(expected_tokens)
    
    def _extract_sql(self, output: str) -> str:
        """Extract SQL from model output."""
        if not output:
            return ""
        
        # Try to find SQL in code blocks
        code_match = re.search(r"```sql\s*(.*?)```", output, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find SELECT statement
        select_match = re.search(r"(SELECT\s+.*)", output, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        return output.strip()

    def _compare_numeric_results(
        self,
        expected_rows: Optional[List[Dict[str, Any]]],
        predicted_rows: Optional[List[Dict[str, Any]]]
    ) -> Tuple[bool, float]:
        """
        Compare numeric values between expected and predicted rows.
        
        Returns:
            Tuple of (match, similarity)
        """
        if expected_rows is None or predicted_rows is None:
            return False, 0.0
        if not expected_rows and not predicted_rows:
            return True, 1.0
        if not expected_rows or not predicted_rows:
            return False, 0.0
        
        sample_row = expected_rows[0]
        numeric_cols = [
            key for key, value in sample_row.items()
            if isinstance(value, (int, float))
        ]
        if not numeric_cols:
            return False, 0.0
        
        non_numeric_cols = [key for key in sample_row.keys() if key not in numeric_cols]
        
        def sort_key(row):
            return tuple(str(row.get(col, "")) for col in non_numeric_cols)
        
        expected_sorted = sorted(expected_rows, key=sort_key)
        predicted_sorted = sorted(predicted_rows, key=sort_key)
        
        total_cells = 0
        correct_cells = 0
        
        for exp_row, pred_row in zip(expected_sorted, predicted_sorted):
            for col in numeric_cols:
                exp_val = exp_row.get(col)
                pred_val = pred_row.get(col)
                if exp_val is None or pred_val is None:
                    continue
                total_cells += 1
                tolerance = self.numeric_tolerance * max(1.0, abs(float(exp_val)))
                if abs(float(pred_val) - float(exp_val)) <= tolerance:
                    correct_cells += 1
        
        similarity = correct_cells / total_cells if total_cells > 0 else 0.0
        match = (len(expected_rows) == len(predicted_rows)) and similarity >= 0.99
        return match, similarity
    
    def predict(self, query: str) -> Tuple[str, str, float]:
        """
        Predict SQL for a single query.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (predicted_sql, raw_output, inference_time_ms)
        """
        self._load_model()
        
        # Format prompt (same as training)
        prompt = f"""Convert the following insurance data query to ClickHouse SQL.
Table: insurance_loss_data
Columns: policy_id, claim_id, state, zip_code, class_code, class_group, coverage_type, 
         policy_year, accident_year, industry, region, written_premium, earned_premium,
         incurred_loss, paid_loss, reserved_loss, exposure_units, claim_count, policy_count

Important: 
- Use SUM/SUM for ratios (NOT AVG of ratios)
- Use NULLIF for division to prevent divide by zero
- Include appropriate GROUP BY for aggregations

{query}"""
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self._model.device)
        
        # Generate
        start_time = time.time()
        
        with __import__("torch").no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Decode
        raw_output = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Extract SQL
        predicted_sql = self._extract_sql(raw_output)
        
        return predicted_sql, raw_output, inference_time
    
    def test_single(self, sample: Dict) -> NL2SQLTestResult:
        """
        Test a single sample.
        
        Args:
            sample: Test sample with 'input' and 'output' keys
            
        Returns:
            NL2SQLTestResult
        """
        query = sample["input"]
        expected_sql = sample["output"]
        ground_truth = sample.get("ground_truth", {})
        
        try:
            predicted_sql, raw_output, inference_time = self.predict(query)
            
            expected_norm = self._normalize_sql(expected_sql)
            predicted_norm = self._normalize_sql(predicted_sql)
            sql_match = expected_norm == predicted_norm
            similarity = self._sql_similarity(expected_sql, predicted_sql)
            
            # Execute predicted SQL to validate syntax and get results
            self._load_executor()
            exec_result = self._executor.execute(predicted_sql)
            execution_success = exec_result.success
            execution_error = exec_result.error
            
            # Compare numeric results if ground truth available
            expected_rows = ground_truth.get("data") if isinstance(ground_truth, dict) else None
            if expected_rows is not None and exec_result.success:
                predicted_rows = exec_result.data
                numeric_match, numeric_similarity = self._compare_numeric_results(expected_rows, predicted_rows)
            else:
                numeric_match, numeric_similarity = False, -1.0
            
            return NL2SQLTestResult(
                query=query,
                expected_sql=expected_sql,
                predicted_sql=predicted_sql,
                sql_match=sql_match,
                sql_similarity=similarity,
                execution_success=execution_success,
                execution_error=execution_error,
                numeric_match=numeric_match,
                numeric_similarity=numeric_similarity,
                inference_time_ms=inference_time,
                raw_output=raw_output,
            )
        except Exception as e:
            logger.error(f"Error testing query: {e}")
            return NL2SQLTestResult(
                query=query,
                expected_sql=expected_sql,
                predicted_sql="",
                sql_match=False,
                sql_similarity=0.0,
                execution_success=False,
                execution_error=str(e),
                numeric_match=False,
                numeric_similarity=0.0,
                inference_time_ms=0,
                raw_output=str(e),
            )
    
    def run_tests(
        self,
        test_file: str,
        limit: Optional[int] = None,
        verbose: bool = True
    ) -> NL2SQLTestReport:
        """
        Run tests from a test file.
        
        Args:
            test_file: Path to test data file (JSONL format)
            limit: Maximum number of tests to run
            verbose: Whether to print progress
            
        Returns:
            NL2SQLTestReport with results
        """
        # Load test data
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
        
        if limit:
            test_samples = test_samples[:limit]
        
        logger.info(f"Running {len(test_samples)} NL2SQL tests...")
        
        report = NL2SQLTestReport()
        report.total_tests = len(test_samples)
        
        similarity_sum = 0.0
        numeric_similarity_sum = 0.0
        
        for i, sample in enumerate(test_samples):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(test_samples)}")
            
            result = self.test_single(sample)
            report.results.append(result)
            
            if result.sql_match:
                report.correct_predictions += 1
            
            similarity_sum += result.sql_similarity
            if result.execution_success:
                report.execution_successes += 1
            if result.numeric_similarity >= 0:
                report.numeric_evaluated += 1
                numeric_similarity_sum += result.numeric_similarity
                if result.numeric_match:
                    report.numeric_correct += 1
            report.total_inference_time_ms += result.inference_time_ms
            
            if not result.predicted_sql:
                report.errors.append(f"{result.query}: No SQL generated")
        
        # Calculate averages
        if report.total_tests > 0:
            report.avg_sql_similarity = similarity_sum / report.total_tests
            report.avg_inference_time_ms = report.total_inference_time_ms / report.total_tests
        if report.numeric_evaluated > 0:
            report.avg_numeric_similarity = numeric_similarity_sum / report.numeric_evaluated
        
        return report
    
    def print_report(self, report: NL2SQLTestReport):
        """Print a formatted test report."""
        print("\n" + "=" * 60)
        print("NL2SQL LORA TEST REPORT")
        print("=" * 60)
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {report.total_tests}")
        print(f"  Correct: {report.correct_predictions}")
        print(f"  Accuracy: {report.accuracy:.2%}")
        
        print(f"\nSQL Similarity:")
        print(f"  Average Similarity: {report.avg_sql_similarity:.2%}")
        
        print(f"\nExecution:")
        print(f"  Success Rate: {report.execution_successes / report.total_tests:.2%}")
        
        if report.numeric_evaluated > 0:
            print(f"\nNumeric Accuracy:")
            print(f"  Evaluated: {report.numeric_evaluated}")
            print(f"  Correct: {report.numeric_correct}")
            print(f"  Accuracy: {report.numeric_correct / report.numeric_evaluated:.2%}")
            print(f"  Avg Similarity: {report.avg_numeric_similarity:.2%}")
        
        print(f"\nPerformance:")
        print(f"  Average Inference Time: {report.avg_inference_time_ms:.2f}ms")
        print(f"  Total Time: {report.total_inference_time_ms:.2f}ms")
        
        if report.errors:
            print(f"\nErrors: {len(report.errors)}")
            for err in report.errors[:5]:
                print(f"  - {err[:80]}...")
        
        print("\n" + "=" * 60)
    
    def save_report(self, report: NL2SQLTestReport, output_path: str):
        """Save report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """CLI entry point for NL2SQL LoRA testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NL2SQL LoRA adapter")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test data file (JSONL format)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to merged model"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/opt/Data_Copilot/models/Qwen2.5-7B-Instructt",
        help="Base model name"
    )
    parser.add_argument(
        "--dataframe-path",
        type=str,
        default=None,
        help="Path to NL2SQL test dataframe (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tests"
    )
    
    args = parser.parse_args()
    
    if not args.model_path and not args.adapter_path:
        parser.error("Must specify either --model-path or --adapter-path")
    
    tester = NL2SQLTester(
        model_path=args.model_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        dataframe_path=args.dataframe_path,
    )
    
    report = tester.run_tests(args.test_file, limit=args.limit)
    tester.print_report(report)
    
    if args.output:
        tester.save_report(report, args.output)


if __name__ == "__main__":
    main()
