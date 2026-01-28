"""
Unified test runner for all LoRA adapters.

Runs tests for:
- Intent LoRA
- Keyword LoRA
- NL2SQL LoRA
"""
import argparse
from pathlib import Path
from loguru import logger

from data_copilot.testing.intent_tester import IntentTester
from data_copilot.testing.keyword_tester import KeywordTester
from data_copilot.testing.nl2sql_tester import NL2SQLTester


def run_intent_tests(args):
    """Run Intent LoRA tests."""
    if not args.intent_model and not args.intent_adapter:
        raise ValueError("Intent tests require --intent-model or --intent-adapter")
    
    tester = IntentTester(
        model_path=args.intent_model,
        base_model=args.base_model,
        adapter_path=args.intent_adapter,
    )
    
    report = tester.run_tests(args.intent_test_file, limit=args.limit)
    tester.print_report(report)
    
    if args.output_dir:
        output_path = Path(args.output_dir) / "intent_report.json"
        tester.save_report(report, str(output_path))


def run_keyword_tests(args):
    """Run Keyword LoRA tests."""
    if not args.keyword_model and not args.keyword_adapter:
        raise ValueError("Keyword tests require --keyword-model or --keyword-adapter")
    
    tester = KeywordTester(
        model_path=args.keyword_model,
        base_model=args.base_model,
        adapter_path=args.keyword_adapter,
    )
    
    report = tester.run_tests(args.keyword_test_file, limit=args.limit)
    tester.print_report(report)
    
    if args.output_dir:
        output_path = Path(args.output_dir) / "keyword_report.json"
        tester.save_report(report, str(output_path))


def run_nl2sql_tests(args):
    """Run NL2SQL LoRA tests."""
    if not args.nl2sql_model and not args.nl2sql_adapter:
        raise ValueError("NL2SQL tests require --nl2sql-model or --nl2sql-adapter")
    
    tester = NL2SQLTester(
        model_path=args.nl2sql_model,
        base_model=args.base_model,
        adapter_path=args.nl2sql_adapter,
        dataframe_path=args.nl2sql_dataframe,
    )
    
    report = tester.run_tests(args.nl2sql_test_file, limit=args.limit)
    tester.print_report(report)
    
    if args.output_dir:
        output_path = Path(args.output_dir) / "nl2sql_report.json"
        tester.save_report(report, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Run LoRA adapter tests")
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["intent", "keyword", "nl2sql", "all"],
        default="all",
        help="Which adapter to test"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/opt/Data_Copilot/models/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tests per adapter"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save test reports"
    )
    
    # Test data files
    parser.add_argument(
        "--intent-test-file",
        type=str,
        default="./data/testing/intent_test.json",
        help="Intent test data file"
    )
    parser.add_argument(
        "--keyword-test-file",
        type=str,
        default="./data/testing/keyword_test.json",
        help="Keyword test data file"
    )
    parser.add_argument(
        "--nl2sql-test-file",
        type=str,
        default="./data/testing/nl2sql_test.json",
        help="NL2SQL test data file"
    )
    parser.add_argument(
        "--nl2sql-dataframe",
        type=str,
        default="./data/testing/nl2sql_test_df.csv",
        help="NL2SQL test dataframe file (CSV)"
    )
    
    # Intent model paths
    parser.add_argument(
        "--intent-model",
        type=str,
        default=None,
        help="Path to merged Intent model"
    )
    parser.add_argument(
        "--intent-adapter",
        type=str,
        default=None,
        help="Path to Intent LoRA adapter"
    )
    
    # Keyword model paths
    parser.add_argument(
        "--keyword-model",
        type=str,
        default=None,
        help="Path to merged Keyword model"
    )
    parser.add_argument(
        "--keyword-adapter",
        type=str,
        default=None,
        help="Path to Keyword LoRA adapter"
    )
    
    # NL2SQL model paths
    parser.add_argument(
        "--nl2sql-model",
        type=str,
        default=None,
        help="Path to merged NL2SQL model"
    )
    parser.add_argument(
        "--nl2sql-adapter",
        type=str,
        default=None,
        help="Path to NL2SQL LoRA adapter"
    )
    
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.adapter in ["intent", "all"]:
            logger.info("Running Intent LoRA tests...")
            run_intent_tests(args)
        
        if args.adapter in ["keyword", "all"]:
            logger.info("Running Keyword LoRA tests...")
            run_keyword_tests(args)
        
        if args.adapter in ["nl2sql", "all"]:
            logger.info("Running NL2SQL LoRA tests...")
            run_nl2sql_tests(args)
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise


if __name__ == "__main__":
    main()
