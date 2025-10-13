#!/bin/bash

set -e

EXISTING_IMAGE_NAME="finetune_whisper_lora:v3"
MIG_DEVICE_UUID="MIG-a45fc4b1-c85e-5a3b-8b3d-79191377ec06" 
FINETUNE_CONTAINER_NAME="lora-finetuning-job" 

mkdir -p ./.cache


docker run \
  -d \
  --name "$FINETUNE_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e "$(pwd)/.cache:/app/.cache" \
  -e HF_HOME=/app/.cache/huggingface \
  -e TORCH_HOME=/app/.cache/torch \
  -e TRITON_CACHE_DIR=/app/.cache/triton \
  -e NUMBA_CACHE_DIR=/app/.cache/numba \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=$MIG_DEVICE_UUID \
  --workdir /app \
  -v "$(pwd)":/app \
  -v "/tmp/viet_bud500":"/tmp/viet_bud500" \
  -v "$(pwd)/.cache:/app/.cache" \
  "$EXISTING_IMAGE_NAME" \
  sleep infinity 

echo "Container '$FINETUNE_CONTAINER_NAME' is running in the background."
