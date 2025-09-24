#!/bin/bash

# Stop the script if any command fails
set -e

# --- Configuration ---
IMAGE_NAME="whisper-lora-finetuner"
FINETUNE_CONTAINER_NAME="lora-finetuning-job"
MERGE_CONTAINER_NAME="lora-merging-job"
CONVERT_CONTAINER_NAME="ct2-conversion-job"
GPU_INDEX=0

# --- Workflow Steps ---

echo "======================================================"
echo " STEP 1: Building the Docker image... "
echo "======================================================"
docker build -t "$IMAGE_NAME" .

echo ""
echo "======================================================"
echo " STEP 2: Starting the LoRA fine-tuning process... "
echo "======================================================"
echo "Container will run in the background. To monitor progress, run:"
echo "docker logs -f $FINETUNE_CONTAINER_NAME"
echo ""

docker run \
  -d \
  --name "$FINETUNE_CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 finetune_lora.py

echo "Waiting for fine-tuning to complete. This may take a long time..."
docker wait "$FINETUNE_CONTAINER_NAME"
echo "Fine-tuning has finished."
docker rm "$FINETUNE_CONTAINER_NAME"

echo ""
echo "======================================================"
echo " STEP 3: Merging the trained adapter into the model... "
echo "======================================================"

docker run \
  --name "$MERGE_CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 merge_lora.py

docker rm "$MERGE_CONTAINER_NAME"

# ======================================================
# <<< NEW STEP ADDED HERE >>>
# ======================================================
echo ""
echo "======================================================"
echo " STEP 4: Converting the merged model to CTranslate2... "
echo "======================================================"

docker run \
  --name "$CONVERT_CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 convert_to_ct2.py

docker rm "$CONVERT_CONTAINER_NAME"
# ======================================================

echo ""
echo "======================================================"
echo " ENTIRE WORKFLOW COMPLETE! "
echo " Your final, faster-whisper-ready model is in the 'faster-whisper-final-model' folder."
echo "======================================================"
