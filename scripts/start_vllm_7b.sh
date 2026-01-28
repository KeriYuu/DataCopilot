#!/bin/bash
# ============================================
# vLLM 7B Multi-LoRA Server Startup Script
# 
# Features:
# - SGMV multi-LoRA hot-swapping (run 3 LoRAs on single GPU)
# - Prefix Caching (cache common prompt prefixes)
# - Tensor Parallel (multi-GPU parallelism)
# ============================================

# Configuration
MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
INTENT_LORA="${INTENT_LORA_PATH:-./output/intent_lora}"
NL2SQL_LORA="${NL2SQL_LORA_PATH:-./output/nl2sql_lora}"
KEYWORD_LORA="${KEYWORD_LORA_PATH:-./output/keyword_lora}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${TP_SIZE_7B:-2}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.8}"

echo "============================================"
echo "Starting vLLM 7B Multi-LoRA Server (SGMV)"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "LoRA Adapters:"
echo "  - Intent: ${INTENT_LORA}"
echo "  - NL2SQL: ${NL2SQL_LORA}"
echo "  - Keyword: ${KEYWORD_LORA}"
echo "Port: ${PORT}"
echo "Tensor Parallel: ${TP_SIZE}"
echo "============================================"

# Start vLLM
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --served-model-name Qwen2.5-7B \
    --enable-lora \
    --lora-modules \
        intent=${INTENT_LORA} \
        nl2sql=${NL2SQL_LORA} \
        keyword=${KEYWORD_LORA} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --trust-remote-code \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len 4096 \
    --dtype half \
    --enable-prefix-caching \
    --disable-log-requests
