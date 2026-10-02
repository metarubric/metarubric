#!/usr/bin/env bash
set -euo pipefail
exec env CUDA_VISIBLE_DEVICES="${ACRE_READER_CUDA_VISIBLE_DEVICES:-0}" "${ACRE_PYTHON_BIN:-python}" -u -m vllm.entrypoints.openai.api_server \
    --model "${ACRE_READER_MODEL_PATH:?ACRE_READER_MODEL_PATH is required}" \
    --served-model-name exam-qwen3-1.7b \
    --tensor-parallel-size 1 --max-model-len 16384 --max-num-batched-tokens 16384 \
    --max-num-seqs 64 --gpu-memory-utilization 0.4 --seed 42 --generation-config vllm \
    --enable-prefix-caching --host 127.0.0.1 --port 19331
