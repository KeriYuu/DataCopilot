"""
Keyword/Entity Extraction LoRA Tester (Enhanced).

Tests the Keyword extraction LoRA adapter AND/OR Baseline model.
Extracts and normalizes entities from natural language queries.
"""
import json
import time
import gc
import torch
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class KeywordTestResult:
    """Result of a single keyword extraction test."""
    query: str
    expected_entities: Dict[str, str]
    predicted_entities: Dict[str, str]
    expected_rewrite: str
    predicted_rewrite: str
    entity_accuracy: float  # Legacy metric (Recall-like)
    rewrite_match: bool
    is_correct: bool
    inference_time_ms: float
    raw_output: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "expected_entities": self.expected_entities,
            "predicted_entities": self.predicted_entities,
            "expected_rewrite": self.expected_rewrite,
            "predicted_rewrite": self.predicted_rewrite,
            "entity_accuracy": self.entity_accuracy,
            "rewrite_match": self.rewrite_match,
            "correct": self.is_correct,
            "time_ms": self.inference_time_ms,
            "raw_output": self.raw_output,
        }


@dataclass
class KeywordTestReport:
    """Aggregated test report for Keyword LoRA."""
    model_name: str = "Unknown"
    total_tests: int = 0
    correct_predictions: int = 0  # Perfect matches (Entities + Rewrite)
    
    # Entity extraction raw counts (for global P/R/F1)
    tp_total: int = 0  # True Positives
    fp_total: int = 0  # False Positives
    fn_total: int = 0  # False Negatives
    
    # Rewrite metrics
    rewrite_matches: int = 0
    
    # Performance
    avg_inference_time_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    
    # Results
    results: List[KeywordTestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Overall exact match accuracy (Strict)."""
        if self.total_tests == 0: return 0.0
        return self.correct_predictions / self.total_tests
    
    @property
    def rewrite_accuracy(self) -> float:
        if self.total_tests == 0: return 0.0
        return self.rewrite_matches / self.total_tests

    def print_detailed_metrics(self):
        """Calculates and prints detailed Precision/Recall/F1 metrics."""
        
        # 1. Global Metrics
        precision = self.tp_total / (self.tp_total + self.fp_total) if (self.tp_total + self.fp_total) > 0 else 0.0
        recall = self.tp_total / (self.tp_total + self.fn_total) if (self.tp_total + self.fn_total) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print("\n" + "=" * 60)
        print(f"KEYWORD TEST REPORT: {self.model_name}")
        print("=" * 60)
        
        print(f"\nSummary:")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Perfect Matches: {self.correct_predictions} (Accuracy: {self.accuracy:.2%})")
        print(f"  Rewrite Accuracy: {self.rewrite_accuracy:.2%}")

        print(f"\nEntity Extraction Global Metrics:")
        print(f"  Precision : {precision:.2%} (Of extracted entities, how many were correct?)")
        print(f"  Recall    : {recall:.2%} (Of actual entities, how many did we find?)")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  (TP: {self.tp_total}, FP: {self.fp_total}, FN: {self.fn_total})")
        
        # 2. Per-Type Metrics
        # We need to aggregate stats from results on the fly for per-type
        type_stats = {} # type -> {'tp': 0, 'fp': 0, 'fn': 0}
        
        for res in self.results:
            exp = res.expected_entities
            pred = res.predicted_entities
            
            # Identify all unique keys
            all_keys = set(exp.keys()) | set(pred.keys())
            
            for key in all_keys:
                # Simplification: The 'key' acts as the entity type (e.g., 'state', 'coverage')
                # In this schema, keys ARE types.
                if key not in type_stats:
                    type_stats[key] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                exp_val = str(exp.get(key, "")).strip().upper()
                pred_val = str(pred.get(key, "")).strip().upper()
                
                if key in exp and key in pred:
                    if exp_val == pred_val:
                        type_stats[key]['tp'] += 1
                    else:
                        # Key match but Value mismatch -> Count as both FP (wrong val) and FN (missed correct val)
                        type_stats[key]['fp'] += 1
                        type_stats[key]['fn'] += 1
                elif key in pred and key not in exp:
                    type_stats[key]['fp'] += 1
                elif key in exp and key not in pred:
                    type_stats[key]['fn'] += 1

        print(f"\nBreakdown by Entity Type:")
        print("-" * 75)
        print(f"{'Entity Type':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Count (TP/Total)'}")
        print("-" * 75)
        
        for etype, stats in sorted(type_stats.items()):
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            total_exp = tp + fn
            
            print(f"{etype:<20} | {p:<10.2%} | {r:<10.2%} | {f:<10.4f} | ({tp}/{total_exp})")
        
        print("-" * 75)
        
        print(f"\nPerformance:")
        print(f"  Avg Inference Time: {self.avg_inference_time_ms:.2f}ms")
        
        if self.errors:
            print(f"\nErrors (First 5):")
            for err in self.errors[:5]:
                print(f"  - {err[:80]}...")
        print("=" * 60)
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "summary": {
                "total_tests": self.total_tests,
                "accuracy": self.accuracy,
                "rewrite_accuracy": self.rewrite_accuracy,
            },
            # ... simplified for saving ...
            "results_count": len(self.results)
        }


class KeywordTester:
    """Tester for Keyword/Entity extraction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "/opt/Data_Copilot/models/Qwen2.5-7B-Instruct",
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
                    torch_dtype="auto" # Fix VRAM issues
                )
            elif self.adapter_path:
                # 2. Base + Adapter (LoRA)
                logger.info(f"Loading LoRA from: {self.adapter_path} (Base: {self.base_model})")
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype="auto" # Fix VRAM issues
                )
                self._model = PeftModel.from_pretrained(base, self.adapter_path)
            else:
                # 3. Baseline Only
                logger.info(f"Loading BASELINE model: {self.base_model}")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype="auto" # Fix VRAM issues
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

    def _parse_output(self, output: str) -> Tuple[Dict[str, str], str]:
        try:
            result = json.loads(output.strip())
            return result.get("entities", {}), result.get("rewritten_query", "")
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*"entities"[^{}]*\}', output, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result.get("entities", {}), result.get("rewritten_query", "")
                except:
                    pass
            return {}, ""
    
    def _calculate_metrics(self, expected: Dict[str, str], predicted: Dict[str, str]) -> Tuple[int, int, int]:
        """Calculate TP, FP, FN for a single sample."""
        tp = 0
        fp = 0
        fn = 0
        
        all_keys = set(expected.keys()) | set(predicted.keys())
        
        for key in all_keys:
            exp_val = str(expected.get(key, "")).strip().upper()
            pred_val = str(predicted.get(key, "")).strip().upper()
            
            if key in expected and key in predicted:
                if exp_val == pred_val:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif key in predicted:
                fp += 1
            elif key in expected:
                fn += 1
                
        return tp, fp, fn

    def predict(self, query: str) -> Tuple[Dict[str, str], str, str, float]:
        self._load_model()
        
        prompt = f"""Extract and normalize entities from the following insurance data query.
Map colloquial terms to schema values:
- State names → 2-letter codes (California → CA)
- Coverage nicknames → full names (WC → Workers Compensation)
- Geographic slang → proper regions (SoCal → CA/West)

Output JSON with 'entities' (field:value pairs) and 'rewritten_query' (normalized query).

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
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        raw_output = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        entities, rewrite = self._parse_output(raw_output)
        
        return entities, rewrite, raw_output, inference_time
    
    def run_tests(self, test_file: str, limit: Optional[int] = None, run_name: str = "Test") -> KeywordTestReport:
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
        
        if limit:
            test_samples = test_samples[:limit]
        
        logger.info(f"Running {len(test_samples)} tests for: {run_name}")
        
        report = KeywordTestReport(model_name=run_name)
        report.total_tests = len(test_samples)
        
        for i, sample in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                logger.info(f"[{run_name}] Progress: {i + 1}/{len(test_samples)}")
            
            result = self.test_single(sample)
            report.results.append(result)
            
            if result.is_correct:
                report.correct_predictions += 1
            
            if result.rewrite_match:
                report.rewrite_matches += 1
            
            # Update Report Global Counters
            tp, fp, fn = self._calculate_metrics(result.expected_entities, result.predicted_entities)
            report.tp_total += tp
            report.fp_total += fp
            report.fn_total += fn
            
            report.total_inference_time_ms += result.inference_time_ms
            
            if not result.is_correct and not result.predicted_entities:
                 report.errors.append(f"{result.query}: No entities extracted or parse error")

        if report.total_tests > 0:
            report.avg_inference_time_ms = report.total_inference_time_ms / report.total_tests
        
        return report

    def test_single(self, sample: Dict) -> KeywordTestResult:
        query = sample["input"]
        
        # Parse expected output
        try:
            expected_output = json.loads(sample["output"])
            expected_entities = expected_output.get("entities", {})
            expected_rewrite = expected_output.get("rewritten_query", "")
        except:
            expected_entities = {}
            expected_rewrite = ""
        
        try:
            pred_entities, pred_rewrite, raw_output, inference_time = self.predict(query)
            
            # Calculate metrics
            tp, fp, fn = self._calculate_metrics(expected_entities, pred_entities)
            
            # Legacy "Entity Accuracy" (Recall-like) for backward compatibility
            total_exp = len(expected_entities)
            entity_acc = tp / total_exp if total_exp > 0 else (1.0 if not pred_entities else 0.0)
            
            # Rewrite match logic (Fuzzy)
            def check_rewrite(exp, pred):
                if not exp: return True
                e_norm = exp.lower().strip()
                p_norm = pred.lower().strip()
                if e_norm == p_norm: return True
                e_words = set(e_norm.split())
                p_words = set(p_norm.split())
                overlap = len(e_words & p_words) / len(e_words) if e_words else 1.0
                return overlap >= 0.8

            rewrite_match = check_rewrite(expected_rewrite, pred_rewrite)
            
            # Overall: No errors (FP=0, FN=0) AND Rewrite matches
            is_correct = (fp == 0 and fn == 0) and rewrite_match
            
            return KeywordTestResult(
                query=query, expected_entities=expected_entities, predicted_entities=pred_entities,
                expected_rewrite=expected_rewrite, predicted_rewrite=pred_rewrite,
                entity_accuracy=entity_acc, rewrite_match=rewrite_match,
                is_correct=is_correct, inference_time_ms=inference_time, raw_output=raw_output,
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return KeywordTestResult(
                query=query, expected_entities=expected_entities, predicted_entities={},
                expected_rewrite=expected_rewrite, predicted_rewrite="",
                entity_accuracy=0.0, rewrite_match=False,
                is_correct=False, inference_time_ms=0, raw_output=str(e),
            )

    def print_report(self, report: KeywordTestReport):
        report.print_detailed_metrics()

    def save_report(self, report: KeywordTestReport, output_path: str):
        path_obj = Path(output_path)
        final_path = path_obj.parent / f"{path_obj.stem}_{report.model_name.replace(' ', '_')}{path_obj.suffix}"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {final_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Keyword LoRA and/or Baseline")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test data")
    parser.add_argument("--base-model", type=str, default="/opt/Data_Copilot/models/Qwen2.5-7B-Instruct", help="Base model path")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--model-path", type=str, default=None, help="Path to merged model")
    
    # New Switch
    parser.add_argument("--test-baseline", action="store_true", help="Also run test on Baseline model")
    
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    # 1. Test Adapter/Merged Model
    if args.adapter_path or args.model_path:
        logger.info(">>> STARTING LoRA/MERGED MODEL TEST")
        tester = KeywordTester(
            model_path=args.model_path,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
        )
        report = tester.run_tests(args.test_file, limit=args.limit, run_name="LoRA_Adapter")
        tester.print_report(report)
        if args.output:
            tester.save_report(report, args.output)
        
        tester.unload_model()
    
    # 2. Test Baseline
    if args.test_baseline or (not args.adapter_path and not args.model_path):
        logger.info("\n>>> STARTING BASELINE MODEL TEST")
        tester_base = KeywordTester(
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