#!/bin/bash

# --- Configuration ---
set -e # Exit immediately if a command fails
IMAGE_NAME="whisper-lora-finetuner"
CONTAINER_NAME="lora-finetuning-job"
GPU_INDEX=0

# --- Workflow ---

echo "======================================================"
echo " STEP 1: Building the Docker image... "
echo "======================================================"
docker build -t "$IMAGE_NAME" .

echo "======================================================"
echo " STEP 2: Starting LoRA fine-tuning in the background... "
echo "======================================================"
echo "To monitor progress, open another terminal and run:"
echo "docker logs -f $CONTAINER_NAME"
echo ""

# Make sure no old container with the same name exists
docker rm "$CONTAINER_NAME" || true

docker run \
  -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 finetune_lora.py

# Wait for the training to complete
echo "Waiting for fine-tuning to finish. This can take a long time."
docker wait "$CONTAINER_NAME"
echo "Fine-tuning complete."

echo ""
echo "======================================================"
echo " STEP 3: Merging the new LoRA adapters... "
echo "======================================================"
docker run --rm \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 merge_adapters.py

echo ""
echo "======================================================"
echo " STEP 4: Converting the final model for faster-whisper... "
echo "======================================================"
docker run --rm \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 convert_model.py

echo ""
echo "======================================================"
echo " Workflow complete! Your final model is in the"
echo " 'faster-whisper-converted-model' directory."
echo "======================================================"
