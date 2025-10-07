#!/bin/bash

# Stop the script if any command fails
set -e

# --- Configuration ---
EXISTING_IMAGE_NAME="dso-nexus.vietinbank.vn/ai_docker/finetune_whisper_lora:v2"
MIG_DEVICE_UUID="PASTE_YOUR_MIG_UUID_HERE" # Make sure this is set correctly

# --- Container Names ---
FINETUNE_CONTAINER_NAME="lora-finetuning-job"
MERGE_CONTAINER_NAME="lora-merging-job"
CONVERT_CONTAINER_NAME="ct2-conversion-job"

# --- Workflow Steps ---

# Check if the user has changed the UUID
if [ "$MIG_DEVICE_UUID" == "PASTE_YOUR_MIG_UUID_HERE" ]; then
  echo "!!! ERROR: Please edit this script and set the 'MIG_DEVICE_UUID' variable."
  exit 1
fi

echo "======================================================"
echo " STEP 1: Starting LoRA fine-tuning (Whisper Large V3 Turbo continued)..."
echo "======================================================"

docker run \
  -d \
  --name "$FINETUNE_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e XDG_CACHE_HOME=/tmp \
  -e TORCH_HOME=/tmp \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=$MIG_DEVICE_UUID \
  --workdir /app \
  -v "$(pwd)":/app \
  -v "/tmp/viet_bud500":"/tmp/viet_bud500" \
  "$EXISTING_IMAGE_NAME" \
  python3 finetune_whisper.py

echo "Waiting for fine-tuning to complete..."
docker wait "$FINETUNE_CONTAINER_NAME"
echo "Fine-tuning has finished."
docker rm "$FINETUNE_CONTAINER_NAME"

echo ""
echo "======================================================"
echo " STEP 2: Merging the trained adapter (with turbo config fixes)..."
echo "======================================================"

docker run \
  -d \
  --name "$MERGE_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e XDG_CACHE_HOME=/tmp \
  -e TORCH_HOME=/tmp \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=$MIG_DEVICE_UUID \
  --workdir /app \
  -v "$(pwd)":/app \
  -v "/tmp/viet_bud500":"/tmp/viet_bud500" \
  "$EXISTING_IMAGE_NAME" \
  python3 merge_adapters.py

echo "Waiting for merging to complete..."
docker wait "$MERGE_CONTAINER_NAME"
echo "Merging has finished."
docker rm "$MERGE_CONTAINER_NAME"

echo ""
echo "======================================================"
echo " STEP 3: Converting the merged model to CTranslate2 (turbo-optimized)..."
echo "======================================================"

docker run \
  -d \
  --name "$CONVERT_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e XDG_CACHE_HOME=/tmp \
  -e TORCH_HOME=/tmp \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=$MIG_DEVICE_UUID \
  --workdir /app \
  -v "$(pwd)":/app \
  -v "/tmp/viet_bud500":"/tmp/viet_bud500" \
  "$EXISTING_IMAGE_NAME" \
  python3 convert_model.py

echo "Waiting for conversion to complete..."
docker wait "$CONVERT_CONTAINER_NAME"
echo "Conversion has finished."
docker rm "$CONVERT_CONTAINER_NAME"

echo ""
echo "======================================================"
echo " ENTIRE WORKFLOW COMPLETE! (Quantization Disabled + Cache Fixes) "
echo "======================================================"
