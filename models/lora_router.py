"""
Multi-LoRA Router for dynamic adapter switching.

Features:
- SGMV (Segmented Gather-Matrix-Vector) via vLLM for multi-LoRA hot-swapping
- Prefix Caching support
- Single 7B base model + 3 LoRA adapters with dynamic switching
"""
import json
import time
from enum import Enum
from typing import Optional, Dict, Any, List
import requests
from loguru import logger

from data_copilot.config import settings


class LoRAType(Enum):
    """
    LoRA adapter types for different tasks.
    
    - INTENT: Intent classification (Router) - question classification
    - NL2SQL: Natural language to SQL conversion - SQL generation
    - KEYWORD: Entity/keyword extraction and rewriting - entity extraction
    - BASE: Base model without LoRA (for general generation)
    """
    INTENT = "intent"
    NL2SQL = "nl2sql"
    KEYWORD = "keyword"
    BASE = "base"


# Mapping of LoRA types to model names in vLLM
# These names must match the vLLM --lora-modules parameter
LORA_MODEL_NAMES: Dict[LoRAType, str] = {
    LoRAType.INTENT: "intent",      # Matches --lora-modules intent=...
    LoRAType.NL2SQL: "nl2sql",      # Matches --lora-modules nl2sql=...
    LoRAType.KEYWORD: "keyword",    # Matches --lora-modules keyword=...
    LoRAType.BASE: "Qwen2.5-7B",    # Base model served-model-name
}


class LoRARouter:
    """
    Multi-tenant LoRA serving router with SGMV.
    
    Instead of deploying 3 separate 7B models, we run one frozen 
    Qwen2.5-7B base and dynamically hot-swap the LoRA adapters 
    (Intent, SQL, Keyword) using SGMV kernels via vLLM.
    
    Features:
    - SGMV multi-LoRA hot-swapping (run 3 LoRAs on single GPU)
    - Prefix Caching (cache system prompts)
    - Unified OpenAI-compatible API
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 60,
        enable_prefix_caching: bool = True
    ):
        """
        Initialize the LoRA router.
        
        Args:
            base_url: vLLM server URL (defaults to settings.vllm_url)
            timeout: Request timeout in seconds
            enable_prefix_caching: Whether to use prefix caching
        """
        self.base_url = base_url or settings.vllm_url
        self.timeout = timeout
        self.enable_prefix_caching = enable_prefix_caching
        self._session = requests.Session()
        
        # Cache for common system prompt prefixes
        self._prefix_cache: Dict[LoRAType, str] = {}
        
    def _get_model_name(self, lora_type: LoRAType) -> str:
        """Get the model name for a LoRA type."""
        return LORA_MODEL_NAMES.get(lora_type, LORA_MODEL_NAMES[LoRAType.BASE])
    
    def generate(
        self,
        prompt: str,
        lora_type: LoRAType,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        use_prefix_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using a specific LoRA adapter.
        
        SGMV automatically switches LoRA weights between requests without model reload.
        
        Args:
            prompt: Input prompt
            lora_type: Which LoRA adapter to use (INTENT/NL2SQL/KEYWORD)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            use_prefix_cache: Whether to leverage prefix caching
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        model_name = self._get_model_name(lora_type)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature if temperature is not None else settings.temperature,
            "stop": stop or ["<|endoftext|>", "<|im_end|>"],
            **kwargs
        }
        
        start_time = time.time()
        
        try:
            response = self._session.post(
                f"{self.base_url}/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = (time.time() - start_time) * 1000
            
            # Log performance metrics
            usage = result.get("usage", {})
            logger.debug(
                f"[SGMV-{lora_type.value}] "
                f"time={elapsed:.1f}ms, "
                f"prompt_tokens={usage.get('prompt_tokens', 'N/A')}, "
                f"completion_tokens={usage.get('completion_tokens', 'N/A')}"
            )
            
            return result["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LoRA generation failed for {lora_type.value}: {e}")
            raise
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        lora_type: LoRAType,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using a specific LoRA adapter.
        
        Supports Prefix Caching - if messages start with same prefix, KV cache is reused.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            lora_type: Which LoRA adapter to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        model_name = self._get_model_name(lora_type)
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature if temperature is not None else settings.temperature,
            **kwargs
        }
        
        start_time = time.time()
        
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = (time.time() - start_time) * 1000
            
            # Check prefix cache hit
            usage = result.get("usage", {})
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            
            logger.debug(
                f"[SGMV-{lora_type.value}] "
                f"time={elapsed:.1f}ms, "
                f"cached_tokens={cached_tokens}"
            )
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LoRA chat generation failed for {lora_type.value}: {e}")
            raise

    async def agenerate(
        self,
        prompt: str,
        lora_type: LoRAType,
        **kwargs
    ) -> str:
        """Async version of generate."""
        import aiohttp
        
        model_name = self._get_model_name(lora_type)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", settings.max_tokens),
            "temperature": kwargs.get("temperature", settings.temperature),
            "stop": kwargs.get("stop", ["<|endoftext|>", "<|im_end|>"]),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                result = await response.json()
                return result["choices"][0]["text"].strip()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check vLLM server health and available LoRA adapters.
        
        Returns:
            Health status and available models
        """
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            models = [m["id"] for m in result.get("data", [])]
            
            return {
                "status": "healthy",
                "available_models": models,
                "sgmv_enabled": len(models) > 1,  # Multiple models means SGMV is enabled
                "prefix_caching": settings.enable_prefix_caching,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global router instance
_router: Optional[LoRARouter] = None


def get_router() -> LoRARouter:
    """Get or create the global LoRA router."""
    global _router
    if _router is None:
        _router = LoRARouter()
    return _router


def set_router(router: LoRARouter) -> None:
    """Set the global LoRA router."""
    global _router
    _router = router
