"""
Shared training configuration for all LoRA adapters.

**Strictly follows the original finance project's parameter design**

Original configuration sources:
- lora/CLASSIFY_PTUNING/src/scripts/train.sh
- lora/NL2SQL_TUNING/src/scripts/train.sh  
- lora/KEYWORDS_TUNING/src/scripts/train.sh
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class LoRATarget(Enum):
    """
    LoRA target modules - strictly follows original project.
    
    Original project only uses q_proj, v_proj (2 projection layers)
    """
    QWEN = ["q_proj", "v_proj"]  # Original project config
    QWEN_FULL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class BaseTrainingConfig:
    """
    Base configuration for LoRA training.
    
    Strictly follows original project defaults:
    - lora_target: q_proj,v_proj
    - lr_scheduler_type: cosine
    - logging_steps: 10
    """
    
    # Model - using Qwen2.5 instead of original Qwen1.5
    model_name_or_path: str = "/opt/Data_Copilot/models/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    
    # LoRA - strictly follows original project
    finetuning_type: str = "lora"
    lora_target: List[str] = field(default_factory=lambda: LoRATarget.QWEN.value)  # Only q_proj,v_proj
    lora_rank: int = 8           # LLaMA-Factory default
    lora_alpha: int = 16         # LLaMA-Factory default (alpha = 2 * rank)
    lora_dropout: float = 0.05
    
    # Training - original project defaults
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Optimization
    optim: str = "adamw_torch"
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Logging - original project config
    logging_steps: int = 10
    
    # Wandb observability
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    # Other
    overwrite_cache: bool = True
    plot_loss: bool = True
    
    # Output
    output_dir: str = "./output"
    
    def to_dict(self) -> dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "finetuning_type": self.finetuning_type,
            "lora_target": ",".join(self.lora_target),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "optim": self.optim,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to,
            "overwrite_cache": self.overwrite_cache,
            "plot_loss": self.plot_loss,
            "output_dir": self.output_dir,
        }


@dataclass
class IntentTrainingConfig(BaseTrainingConfig):
    """
    Configuration for Intent classification LoRA.
    
    Strictly follows: lora/CLASSIFY_PTUNING/src/scripts/train.sh
    - learning_rate: 5e-5
    - num_train_epochs: 3
    - save_steps: 20
    """
    
    # Intent-specific (strictly follows original project)
    output_dir: str = "./output/intent_lora"
    learning_rate: float = 5e-5      # Original: 5e-5
    num_train_epochs: int = 3        # Original: 3
    save_steps: int = 20             # Original: 20
    eval_steps: int = 20
    max_length: int = 512
    
    # Dataset
    dataset: str = "intent_train"
    template: str = "qwen" 
    cutoff_len: int = 512
    
    # Wandb
    run_name: str = "intent-lora-v1"


@dataclass 
class NL2SQLTrainingConfig(BaseTrainingConfig):
    """
    Configuration for NL2SQL generation LoRA.
    
    Strictly follows: lora/NL2SQL_TUNING/src/scripts/train.sh
    - learning_rate: 1e-4
    - num_train_epochs: 5
    - save_steps: 50
    - gradient_accumulation_steps: 16
    - per_device_train_batch_size: 1
    """
    
    # NL2SQL-specific (strictly follows original project)
    output_dir: str = "./output/nl2sql_lora"
    learning_rate: float = 1e-4      # Original: 1e-4
    num_train_epochs: int = 5        # Original: 5
    save_steps: int = 50             # Original: 50
    eval_steps: int = 50
    max_length: int = 2048
    
    # Original uses larger gradient accumulation to compensate for small batch
    per_device_train_batch_size: int = 1    # Original: 1
    gradient_accumulation_steps: int = 16   # Original: 16
    
    # Dataset
    dataset: str = "nl2sql_train"
    template: str = "default"
    cutoff_len: int = 2048
    
    # Wandb
    run_name: str = "nl2sql-lora-v1"


@dataclass
class KeywordTrainingConfig(BaseTrainingConfig):
    """
    Configuration for Keyword extraction LoRA.
    
    Strictly follows: lora/KEYWORDS_TUNING/src/scripts/train.sh
    - learning_rate: 5e-5
    - num_train_epochs: 3
    - save_steps: 100
    """
    
    # Keyword-specific (strictly follows original project)
    output_dir: str = "./output/keyword_lora"
    learning_rate: float = 5e-5      # Original: 5e-5
    num_train_epochs: int = 3        # Original: 3
    save_steps: int = 100            # Original: 100
    eval_steps: int = 100
    max_length: int = 512
    
    # Dataset
    dataset: str = "keyword_train"
    template: str = "default"
    cutoff_len: int = 512
    
    # Wandb
    run_name: str = "keyword-lora-v1"


def get_training_config(adapter_type: str) -> BaseTrainingConfig:
    """Get training configuration for adapter type."""
    configs = {
        "intent": IntentTrainingConfig(),
        "nl2sql": NL2SQLTrainingConfig(),
        "keyword": KeywordTrainingConfig(),
    }
    return configs.get(adapter_type, BaseTrainingConfig())


# Original project parameter reference table
ORIGINAL_PROJECT_PARAMS = """
+===================================================================+
|           Original Finance Project LoRA Training Params           |
+===============+===============+===============+===================+
|    Param      |    Intent     |    NL2SQL     |    Keyword        |
+===============+===============+===============+===================+
| lora_target   | q_proj,v_proj | q_proj,v_proj | q_proj,v_proj     |
| learning_rate |     5e-5      |     1e-4      |     5e-5          |
| epochs        |       3       |       5       |       3           |
| batch_size    |       4       |       1       |       4           |
| grad_accum    |       1       |      16       |       1           |
| save_steps    |      20       |      50       |      100          |
| lr_scheduler  |    cosine     |    cosine     |    cosine         |
+===============+===============+===============+===================+
"""
