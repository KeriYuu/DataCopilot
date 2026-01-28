#!/usr/bin/env python3
"""
Training script for all LoRA adapters.

This script:
1. Generates training data for each adapter (with optional 15% test split)
2. Trains Intent, NL2SQL, and Keyword LoRA adapters
3. Saves adapters for deployment

Usage:
    # Generate data and training scripts for all adapters (with 15% test split)
    python -m training.scripts.train_all --all
    
    # Generate data only (no training scripts)
    python -m training.scripts.train_all --generate-only
    
    # Specify custom test ratio (e.g., 20%)
    python -m training.scripts.train_all --test-ratio 0.20
    
    # Disable test split (training data only)
    python -m training.scripts.train_all --no-test-split
    
    # Train specific adapter
    python -m training.scripts.train_all --adapter intent
    python -m training.scripts.train_all --adapter nl2sql
    python -m training.scripts.train_all --adapter keyword
    
    # With wandb tracking
    python -m training.scripts.train_all --adapter intent --wandb-project data-copilot
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from training.data.intent_generator import IntentDataGenerator
from training.data.nl2sql_generator import NL2SQLDataGenerator
from training.data.keyword_generator import KeywordDataGenerator
from training.scripts.train_config import (
    IntentTrainingConfig,
    NL2SQLTrainingConfig,
    KeywordTrainingConfig,
    ORIGINAL_PROJECT_PARAMS,
)


def setup_wandb(project: str, run_name: str):
    """Setup wandb for experiment tracking."""
    try:
        import wandb
        wandb.init(
            project=project,
            name=run_name,
            config={
                "framework": "LLaMA-Factory",
                "task": "LoRA-SFT",
            }
        )
        logger.info(f"Wandb initialized: project={project}, run={run_name}")
        return True
    except ImportError:
        logger.warning("wandb not installed. Run: pip install wandb")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return False


def generate_training_data(data_dir: str, test_dir: str = None, test_ratio: float = 0.15):
    """
    Generate all training and test datasets.
    
    Sample distribution based on task difficulty:
    - Intent: 800 samples (simple classification)
    - Keyword: 2000 samples (medium - entity extraction)
    - NL2SQL: 4000 samples (hard - SQL generation)
    
    Args:
        data_dir: Directory for training data
        test_dir: Directory for test data (if None, no test data is generated)
        test_ratio: Ratio of test samples (default 15%)
    """
    os.makedirs(data_dir, exist_ok=True)
    if test_dir:
        os.makedirs(test_dir, exist_ok=True)
    
    # Intent data (simplified 2-class: DATA_QUERY, METADATA)
    # Simple task: 800 samples total
    logger.info("Generating intent classification data (800 samples)...")
    intent_gen = IntentDataGenerator()
    
    if test_dir:
        intent_gen.save_train_test_split(
            train_filepath=os.path.join(data_dir, "intent_train.jsonl"),
            test_filepath=os.path.join(test_dir, "intent_test.jsonl"),
            n_data_query=550,   # ~70% - more common query type
            n_metadata=250,     # ~30% - less frequent
            test_ratio=test_ratio
        )
    else:
        intent_gen.save_dataset(
            os.path.join(data_dir, "intent_train.jsonl"),
            n_data_query=550,
            n_metadata=250
        )
    
    # Keyword data (entity extraction and normalization)
    # Medium difficulty: 2000 samples total
    # Harder subtasks get more samples
    logger.info("Generating keyword extraction data (2000 samples)...")
    keyword_gen = KeywordDataGenerator()
    
    if test_dir:
        keyword_gen.save_train_test_split(
            train_filepath=os.path.join(data_dir, "keyword_train.jsonl"),
            test_filepath=os.path.join(test_dir, "keyword_test.jsonl"),
            n_state=200,        # simple - least samples
            n_coverage=250,     # simple
            n_geo=350,          # medium
            n_class=450,        # medium
            n_combined=750,     # hard - most samples
            test_ratio=test_ratio
        )
    else:
        keyword_gen.save_dataset(
            os.path.join(data_dir, "keyword_train.jsonl"),
            n_state=200,
            n_coverage=250,
            n_geo=350,
            n_class=450,
            n_combined=750
        )
    
    # NL2SQL data (natural language to SQL)
    # Hardest task: 4000 samples total
    # Harder subtasks get more samples
    logger.info("Generating NL2SQL data (4000 samples)...")
    nl2sql_gen = NL2SQLDataGenerator()
    
    if test_dir:
        nl2sql_gen.save_train_test_split_with_ground_truth(
            train_filepath=os.path.join(data_dir, "nl2sql_train.jsonl"),
            test_filepath=os.path.join(test_dir, "nl2sql_test.jsonl"),
            dataframe_filepath=os.path.join(test_dir, "nl2sql_test_df.csv"),
            n_pattern=50,       # basic - ~950 samples (50 × 19 patterns)
            n_formula=1450,     # medium - formula calculations
            n_complex=1600,     # hard - most samples
            test_ratio=test_ratio
        )
    else:
        nl2sql_gen.save_dataset(
            os.path.join(data_dir, "nl2sql_train.jsonl"),
            n_pattern=50,
            n_formula=1450,
            n_complex=1600
        )
    
    logger.info(f"Training data generated in {data_dir}")
    if test_dir:
        logger.info(f"Test data generated in {test_dir}")
        logger.info(f"Train/Test split: {100*(1-test_ratio):.0f}%/{100*test_ratio:.0f}%")
    logger.info("Summary: Intent=800, Keyword=2000, NL2SQL=4000 (total before split)")


def create_dataset_info(data_dir: str):
    """Create dataset_info.json for LLaMA-Factory."""
    dataset_info = {
        "intent_train": {
            "file_name": "intent_train.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "nl2sql_train": {
            "file_name": "nl2sql_train.jsonl", 
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "keyword_train": {
            "file_name": "keyword_train.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    
    with open(os.path.join(data_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info("Created dataset_info.json")


def train_adapter(
    adapter_type: str, 
    data_dir: str, 
    output_dir: str,
    wandb_project: str = None
):
    """
    Train a specific LoRA adapter.
    
    Strictly follows the original finance project's training parameters.
    
    Args:
        adapter_type: One of "intent", "nl2sql", "keyword"
        data_dir: Directory containing training data
        output_dir: Directory for model output
        wandb_project: Wandb project name (optional)
    """
    logger.info(f"Training {adapter_type} adapter...")
    logger.info(ORIGINAL_PROJECT_PARAMS)
    
    # Get config
    if adapter_type == "intent":
        config = IntentTrainingConfig()
        config.output_dir = os.path.join(output_dir, "intent_lora")
    elif adapter_type == "nl2sql":
        config = NL2SQLTrainingConfig()
        config.output_dir = os.path.join(output_dir, "nl2sql_lora")
    elif adapter_type == "keyword":
        config = KeywordTrainingConfig()
        config.output_dir = os.path.join(output_dir, "keyword_lora")
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    # Create training command for LLaMA-Factory
    # Strictly follows original project format
    cmd_parts = [
        'CUDA_VISIBLE_DEVICES=0 python -c "$CMD"',
        "--stage sft",
        "--do_train",
        f"--model_name_or_path {config.model_name_or_path}",
        f"--dataset_dir {data_dir}",
        f"--dataset {adapter_type}_train",
        f"--template {config.template}",
        "--finetuning_type lora",
        f"--lora_target {','.join(config.lora_target)}",
        f"--output_dir {config.output_dir}",
        "--overwrite_cache",
        f"--per_device_train_batch_size {config.per_device_train_batch_size}",
        f"--gradient_accumulation_steps {config.gradient_accumulation_steps}",
        f"--lr_scheduler_type {config.lr_scheduler_type}",
        f"--logging_steps {config.logging_steps}",
        f"--save_steps {config.save_steps}",
        f"--learning_rate {config.learning_rate}",
        f"--num_train_epochs {config.num_train_epochs}",
        f"--max_length {config.max_length}",
        "--plot_loss",
    ]
    
    # Add wandb config
    if wandb_project:
        cmd_parts.extend([
            f"--report_to wandb",
            f"--run_name {config.run_name}",
        ])
        # Set wandb environment variable
        os.environ["WANDB_PROJECT"] = wandb_project
    
    # Add bf16
    if config.bf16:
        cmd_parts.append("--bf16 true")
    
    cmd = " \\\n    ".join(cmd_parts)
    
    # Save as shell script
    script_path = os.path.join(output_dir, f"train_{adapter_type}.sh")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Training script for {adapter_type} LoRA adapter\n")
        f.write('CMD="import sys; from llamafactory.cli import main; sys.argv=[\'llamafactory-cli\', \'train\'] + sys.argv[1:]; main()"\n\n')
        
        # Wandb setup
        if wandb_project:
            f.write(f"# Wandb configuration\n")
            f.write(f"export WANDB_PROJECT={wandb_project}\n")
            f.write(f"export WANDB_RUN_NAME={config.run_name}\n\n")
        
        f.write(cmd + "\n")
    
    os.chmod(script_path, 0o755)
    logger.info(f"Training script saved to {script_path}")
    
    # Print config summary
    logger.info(f"""
+==============================================================+
|  {adapter_type.upper()} LoRA Training Config (follows original project)
+==============================================================+
|  lora_target:        {','.join(config.lora_target)}
|  learning_rate:      {config.learning_rate}
|  num_epochs:         {config.num_train_epochs}
|  batch_size:         {config.per_device_train_batch_size}
|  gradient_accum:     {config.gradient_accumulation_steps}
|  save_steps:         {config.save_steps}
|  lr_scheduler:       {config.lr_scheduler_type}
|  wandb_project:      {wandb_project or 'disabled'}
+==============================================================+
""")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters (follows original finance project)")
    parser.add_argument(
        "--adapter", 
        type=str, 
        choices=["intent", "nl2sql", "keyword", "all"],
        default="all",
        help="Which adapter to train"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/training",
        help="Directory for training data"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="./data/testing",
        help="Directory for test data"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of test samples (default 15%%)"
    )
    parser.add_argument(
        "--no-test-split",
        action="store_true",
        help="Disable train/test split (only generate training data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for model output"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate training data, don't train"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name for tracking (e.g., 'data-copilot')"
    )
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Show original project parameters and exit"
    )
    
    args = parser.parse_args()
    
    if args.show_params:
        print(ORIGINAL_PROJECT_PARAMS)
        return
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine test directory
    test_dir = None if args.no_test_split else args.test_dir
    
    # Generate training data (and test data if enabled)
    generate_training_data(
        data_dir=args.data_dir,
        test_dir=test_dir,
        test_ratio=args.test_ratio
    )
    create_dataset_info(args.data_dir)
    
    if args.generate_only:
        logger.info("Data generation complete. Skipping training.")
        return
    
    # Train adapters
    adapters = ["intent", "nl2sql", "keyword"] if args.adapter == "all" else [args.adapter]
    
    for adapter in adapters:
        script = train_adapter(
            adapter, 
            args.data_dir, 
            args.output_dir,
            wandb_project=args.wandb_project
        )
        logger.info(f"Run training with: bash {script}")


if __name__ == "__main__":
    main()
