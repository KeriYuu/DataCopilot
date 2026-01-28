"""
Configuration settings for Data Copilot.

Uses pydantic-settings for environment variable support.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
            env_file=".env", 
            env_file_encoding="utf-8",
            extra="ignore"
        )
    
    # ========================================
    # Database settings
    # ========================================
    clickhouse_host: str = Field(default="localhost") 
    clickhouse_port: int = Field(default=8123)
    clickhouse_database: str = Field(default="insurance_db")
    clickhouse_user: str = Field(default="default")
    clickhouse_password: str = Field(default="")
    
    # ========================================
    # vLLM 7B + Multi-LoRA Service (SGMV)
    # ========================================
    vllm_url: str = Field(
        default="http://localhost:8000/v1", 
        env="VLLM_URL",
        description="vLLM server URL for 7B model with multiple LoRA adapters"
    )
    vllm_port: int = Field(
        default=8000,
        env="VLLM_PORT",
        description="vLLM server port for 7B model (used by shell scripts)"
    )
    
    # Base model
    base_model_path: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        env="BASE_MODEL_PATH"
    )
    
    # LoRA adapter paths
    intent_lora_path: str = Field(default="./output/intent_lora", env="INTENT_LORA_PATH")
    nl2sql_lora_path: str = Field(default="./output/nl2sql_lora", env="NL2SQL_LORA_PATH")
    keyword_lora_path: str = Field(default="./output/keyword_lora", env="KEYWORD_LORA_PATH")
    
    # ========================================
    # vLLM 32B Service (GPTQ Quantized)
    # ========================================
    vllm_32b_url: str = Field(
        default="http://localhost:8001/v1",
        env="VLLM_32B_URL",
        description="vLLM server URL for 32B GPTQ quantized model"
    )
    vllm_32b_port: int = Field(
        default=8001,
        env="VLLM_32B_PORT",
        description="vLLM server port for 32B model (used by shell scripts)"
    )
    
    generator_model: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",  # GPTQ quantized version
        env="GENERATOR_MODEL",
        description="32B model with GPTQ quantization"
    )
    
    # ========================================
    # Inference Optimization Config
    # ========================================
    
    # Prefix Caching (cache common prompt prefixes)
    enable_prefix_caching: bool = Field(
        default=True,
        env="ENABLE_PREFIX_CACHING",
        description="Enable vLLM prefix caching for multi-turn conversations"
    )
    
    # Speculative Decoding (use 7B to accelerate 32B)
    enable_speculative_decoding: bool = Field(
        default=True,
        env="ENABLE_SPECULATIVE_DECODING",
        description="Use 7B as draft model for 32B speculative decoding"
    )
    speculative_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        env="SPECULATIVE_MODEL",
        description="Draft model for speculative decoding"
    )
    num_speculative_tokens: int = Field(
        default=5,
        env="NUM_SPECULATIVE_TOKENS",
        description="Number of tokens to speculate"
    )
    
    # Tensor Parallel
    tensor_parallel_size_7b: int = Field(default=2, validation_alias="TP_SIZE_7B")
    tensor_parallel_size_32b: int = Field(default=4, validation_alias="TP_SIZE_32B")
    
    # GPU Memory
    gpu_memory_utilization: float = Field(default=0.8)
    
    # ========================================
    # Generation settings
    # ========================================
    max_tokens: int = Field(default=1024, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_sql_retries: int = Field(default=3, env="MAX_SQL_RETRIES")
    
    # ========================================
    # MLflow settings
    # ========================================
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="data_copilot", env="MLFLOW_EXPERIMENT_NAME")
    
    # ========================================
    # API settings
    # ========================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    


# Global settings instance
settings = Settings()


# ========================================
# vLLM Startup Command Generation
# ========================================

def get_vllm_7b_command() -> str:
    """
    Generate 7B multi-LoRA vLLM startup command.
    
    Features:
    - SGMV multi-LoRA hot-swapping
    - Prefix Caching
    - Tensor Parallel
    """
    cmd_parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--host 0.0.0.0",
        f"--port {settings.vllm_port}",
        f"--model {settings.base_model_path}",
        "--served-model-name Qwen2.5-7B",
        "--enable-lora",  # Enable multi-LoRA (SGMV)
        f"--lora-modules intent={settings.intent_lora_path} "
        f"nl2sql={settings.nl2sql_lora_path} "
        f"keyword={settings.keyword_lora_path}",
        f"--gpu-memory-utilization {settings.gpu_memory_utilization}",
        "--trust-remote-code",
        f"--tensor-parallel-size {settings.tensor_parallel_size_7b}",
        "--max-model-len 4096",
        "--dtype half",
    ]
    
    # Prefix caching
    if settings.enable_prefix_caching:
        cmd_parts.append("--enable-prefix-caching")
    
    return " \\\n    ".join(cmd_parts)


def get_vllm_32b_command() -> str:
    """
    Generate 32B GPTQ vLLM startup command.
    
    Features:
    - GPTQ INT8 quantization
    - Speculative decoding (optional)
    - Tensor Parallel
    - Prefix Caching
    """
    cmd_parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--host 0.0.0.0",
        f"--port {settings.vllm_32b_port}",
        f"--model {settings.generator_model}",
        "--served-model-name Qwen2.5-32B-GPTQ",
        "--quantization gptq",  # GPTQ quantization
        f"--gpu-memory-utilization {settings.gpu_memory_utilization}",
        "--trust-remote-code",
        f"--tensor-parallel-size {settings.tensor_parallel_size_32b}",
        "--max-model-len 8192",
        "--dtype half",
    ]
    
    # Prefix caching
    if settings.enable_prefix_caching:
        cmd_parts.append("--enable-prefix-caching")
    
    # Speculative decoding
    if settings.enable_speculative_decoding:
        cmd_parts.extend([
            f"--speculative-model {settings.speculative_model}",
            f"--num-speculative-tokens {settings.num_speculative_tokens}",
            "--use-v2-block-manager",  # Required for speculative decoding
        ])
    
    return " \\\n    ".join(cmd_parts)


def print_vllm_commands():
    """Print vLLM startup commands."""
    print("=" * 70)
    print("vLLM 7B Multi-LoRA Service (SGMV + Prefix Caching)")
    print("=" * 70)
    print(get_vllm_7b_command())
    print()
    print("=" * 70)
    print("vLLM 32B GPTQ Service (Quantization + Speculative Decoding + Prefix Caching)")
    print("=" * 70)
    print(get_vllm_32b_command())


if __name__ == "__main__":
    print_vllm_commands()
