#!/bin/bash
# ============================================
# vLLM 32B GPTQ Server Startup Script
# 
# Features:
# - GPTQ INT8 quantization (~16GB VRAM)
# - Speculative decoding (7B as draft model, ~2x speedup)
# - Prefix Caching
# - Tensor Parallel (4 GPUs)
# ============================================

# Configuration
MODEL_PATH="${GENERATOR_MODEL:-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8}"
DRAFT_MODEL="${SPECULATIVE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_32B_PORT:-8001}"
TP_SIZE="${TP_SIZE_32B:-4}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.8}"
NUM_SPEC_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
ENABLE_SPEC="${ENABLE_SPECULATIVE_DECODING:-true}"

echo "============================================"
echo "Starting vLLM 32B GPTQ Server"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Quantization: GPTQ INT8"
echo "Port: ${PORT}"
echo "Tensor Parallel: ${TP_SIZE}"
echo "Speculative Decoding: ${ENABLE_SPEC}"
if [ "${ENABLE_SPEC}" = "true" ]; then
    echo "  Draft Model: ${DRAFT_MODEL}"
    echo "  Speculative Tokens: ${NUM_SPEC_TOKENS}"
fi
echo "============================================"

# Base command
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --served-model-name Qwen2.5-32B-GPTQ \
    --quantization gptq \
    --gpu-memory-utilization ${GPU_UTIL} \
    --trust-remote-code \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len 8192 \
    --dtype half \
    --enable-prefix-caching \
    --disable-log-requests"

# Add speculative decoding configuration
if [ "${ENABLE_SPEC}" = "true" ]; then
    CMD="${CMD} \
    --speculative-model ${DRAFT_MODEL} \
    --num-speculative-tokens ${NUM_SPEC_TOKENS} \
    --use-v2-block-manager"
fi

# Execute
eval ${CMD}
