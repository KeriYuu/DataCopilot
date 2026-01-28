"""
Intent LoRA Tester (Enhanced).

Tests the Intent classification LoRA adapter AND/OR Baseline model.
Classifies queries into:
- DATA_QUERY: Database queries (lookups, aggregations, calculations)
- METADATA: Definitions, system info, general questions
"""
import json
import time
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import torch


@dataclass
class IntentTestResult:
    """Result of a single intent classification test."""
    query: str
    expected_intent: str
    predicted_intent: str
    is_correct: bool
    inference_time_ms: float
    raw_output: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "expected": self.expected_intent,
            "predicted": self.predicted_intent,
            "correct": self.is_correct,
            "time_ms": self.inference_time_ms,
            "raw_output": self.raw_output,
        }


@dataclass
class IntentTestReport:
    """Aggregated test report for Intent LoRA."""
    model_name: str = "Unknown"
    total_tests: int = 0
    correct_predictions: int = 0
    
    # By category
    data_query_total: int = 0
    data_query_correct: int = 0
    metadata_total: int = 0
    metadata_correct: int = 0
    
    # Performance
    avg_inference_time_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    
    # Results
    results: List[IntentTestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total_tests == 0:
            return 0.0
        return self.correct_predictions / self.total_tests
    
    @property
    def data_query_accuracy(self) -> float:
        if self.data_query_total == 0:
            return 0.0
        return self.data_query_correct / self.data_query_total
    
    @property
    def metadata_accuracy(self) -> float:
        if self.metadata_total == 0:
            return 0.0
        return self.metadata_correct / self.metadata_total
    
    def print_confusion_matrix(self):
        """Calculates and prints the Confusion Matrix and Classification Report."""
        # Define vectors
        # Positive Class = DATA_QUERY
        # Negative Class = METADATA
        
        tp = 0  # True Positive (Exp: DATA_QUERY, Pred: DATA_QUERY)
        fn = 0  # False Negative (Exp: DATA_QUERY, Pred: METADATA)
        fp = 0  # False Positive (Exp: METADATA, Pred: DATA_QUERY)
        tn = 0  # True Negative (Exp: METADATA, Pred: METADATA)
        
        for r in self.results:
            exp = r.expected_intent
            pred = r.predicted_intent
            
            if exp == "DATA_QUERY":
                if pred == "DATA_QUERY":
                    tp += 1
                else:
                    fn += 1 # Predicted METADATA or ERROR
            elif exp == "METADATA":
                if pred == "METADATA":
                    tn += 1
                else:
                    fp += 1 # Predicted DATA_QUERY or ERROR

        # Calculate Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n{self.model_name} - Confusion Matrix:")
        print("-" * 60)
        print(f"{'':>20} | {'Pred: DATA_QUERY':>18} | {'Pred: METADATA':>16}")
        print("-" * 60)
        print(f"{'Actual: DATA_QUERY':>20} | {tp:>18} (TP) | {fn:>16} (FN)")
        print(f"{'Actual: METADATA':>20} | {fp:>18} (FP) | {tn:>16} (TN)")
        print("-" * 60)
        
        print(f"\nDetailed Metrics (Target Class: DATA_QUERY):")
        print(f"  Precision : {precision:.2%} (Of all detected queries, how many were actual queries?)")
        print(f"  Recall    : {recall:.2%}    (Of all actual queries, how many did we find?)")
        print(f"  F1-Score  : {f1:.4f}")


class IntentTester:
    """Tester for Intent classification."""
    
    VALID_INTENTS = {"DATA_QUERY", "METADATA"}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = device
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load model and tokenizer (lazy loading)."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Load tokenizer
            # If checking baseline, model_path is None, use base_model
            name_to_load = self.model_path or self.base_model
            logger.info(f"Loading Tokenizer from: {name_to_load}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                name_to_load,
                trust_remote_code=True
            )
            
            # --- MODEL LOADING LOGIC ---
            if self.model_path:
                # 1. Merged Model
                logger.info(f"Loading MERGED model from: {self.model_path}")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype="auto"  # Fix for VRAM usage
                )
            elif self.adapter_path:
                # 2. Base + Adapter (LoRA)
                logger.info(f"Loading LoRA from: {self.adapter_path} (Base: {self.base_model})")
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype="auto"  # Fix for VRAM usage
                )
                self._model = PeftModel.from_pretrained(base, self.adapter_path)
            else:
                # 3. Baseline Only (No Adapter)
                logger.info(f"Loading BASELINE model: {self.base_model}")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype="auto"  # Fix for VRAM usage
                )
            
            self._model.eval()
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise

    def unload_model(self):
        """Unload model to free VRAM for next test."""
        if self._model is not None:
            logger.info("Unloading model to free VRAM...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def _extract_intent(self, output: str) -> str:
        output_upper = output.upper().strip()
        for intent in self.VALID_INTENTS:
            if intent in output_upper:
                return intent
        if "DATA" in output_upper and "QUERY" in output_upper:
            return "DATA_QUERY"
        if "META" in output_upper:
            return "METADATA"
        return "DATA_QUERY"
    
    def predict(self, query: str) -> Tuple[str, str, float]:
        self._load_model()
        
        # Consistent Prompting
        prompt = f"""Classify the following insurance data query into one of two categories: DATA_QUERY (any database query including lookups, aggregations, calculations) or METADATA (definitions, system info, general questions). Respond with ONLY the category name.

{query}"""
        
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self._model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        raw_output = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        predicted = self._extract_intent(raw_output)
        return predicted, raw_output, inference_time
    
    def run_tests(self, test_file: str, limit: Optional[int] = None, run_name: str = "Test") -> IntentTestReport:
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
        
        if limit:
            test_samples = test_samples[:limit]
        
        logger.info(f"Running {len(test_samples)} tests for: {run_name}")
        
        report = IntentTestReport(model_name=run_name)
        report.total_tests = len(test_samples)
        
        for i, sample in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                logger.info(f"[{run_name}] Progress: {i + 1}/{len(test_samples)}")
            
            result = self.test_single(sample)
            report.results.append(result)
            
            if result.is_correct:
                report.correct_predictions += 1
            
            if result.expected_intent == "DATA_QUERY":
                report.data_query_total += 1
                if result.is_correct:
                    report.data_query_correct += 1
            elif result.expected_intent == "METADATA":
                report.metadata_total += 1
                if result.is_correct:
                    report.metadata_correct += 1
            
            report.total_inference_time_ms += result.inference_time_ms
            
            if result.predicted_intent == "ERROR":
                report.errors.append(f"{result.query}: {result.raw_output}")
        
        if report.total_tests > 0:
            report.avg_inference_time_ms = report.total_inference_time_ms / report.total_tests
            
        return report

    def test_single(self, sample: Dict) -> IntentTestResult:
        query = sample["input"]
        expected = sample["output"]
        try:
            predicted, raw_output, inference_time = self.predict(query)
            is_correct = predicted == expected
            return IntentTestResult(
                query=query, expected_intent=expected, predicted_intent=predicted,
                is_correct=is_correct, inference_time_ms=inference_time, raw_output=raw_output
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return IntentTestResult(
                query=query, expected_intent=expected, predicted_intent="ERROR",
                is_correct=False, inference_time_ms=0, raw_output=str(e)
            )

    def print_report(self, report: IntentTestReport):
        print("\n" + "=" * 60)
        print(f"INTENT TEST REPORT: {report.model_name}")
        print("=" * 60)
        print(f"  Accuracy: {report.accuracy:.2%} ({report.correct_predictions}/{report.total_tests})")
        print(f"  DATA_QUERY Acc: {report.data_query_accuracy:.2%}")
        print(f"  METADATA Acc:   {report.metadata_accuracy:.2%}")
        
        # Print Confusion Matrix
        report.print_confusion_matrix()
        print("\n" + "=" * 60)

    def save_report(self, report: IntentTestReport, output_path: str):
        # Insert model name into filename if saving
        path_obj = Path(output_path)
        final_path = path_obj.parent / f"{path_obj.stem}_{report.model_name.replace(' ', '_')}{path_obj.suffix}"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {final_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Intent LoRA and/or Baseline")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test data")
    parser.add_argument("--base-model", type=str, default="/opt/Data_Copilot/models/Qwen2.5-7B-Instruct", help="Base model path")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--model-path", type=str, default=None, help="Path to merged model")
    
    # New Switch
    parser.add_argument("--test-baseline", action="store_true", help="Also run test on Baseline model")
    
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    # 1. Test Adapter/Merged Model (if provided)
    if args.adapter_path or args.model_path:
        logger.info(">>> STARTING LoRA/MERGED MODEL TEST")
        tester = IntentTester(
            model_path=args.model_path,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
        )
        report = tester.run_tests(args.test_file, limit=args.limit, run_name="LoRA_Adapter")
        tester.print_report(report)
        if args.output:
            tester.save_report(report, args.output)
        
        # Cleanup before next test
        tester.unload_model()
    
    # 2. Test Baseline (if requested OR if no adapter provided)
    if args.test_baseline or (not args.adapter_path and not args.model_path):
        logger.info("\n>>> STARTING BASELINE MODEL TEST")
        # Initialize with NO adapter and NO merged model path
        tester_base = IntentTester(
            model_path=None,
            base_model=args.base_model,
            adapter_path=None
        )
        report_base = tester_base.run_tests(args.test_file, limit=args.limit, run_name="Baseline")
        tester_base.print_report(report_base)
        if args.output:
            tester_base.save_report(report_base, args.output)

if __name__ == "__main__":
    main()