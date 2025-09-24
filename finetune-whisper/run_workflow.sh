#!/bin/bash

# --- Configuration ---
# Stop the script if any command fails
set -e

# SET YOUR PRE-BUILT IMAGE NAME HERE
IMAGE_NAME="my-whisper-environment:1.0"

# SET THE NAME FOR YOUR DOCKER CONTAINER (for easy management)
CONTAINER_NAME="whisper-finetuning-job"

# SET THE GPU INDEX YOU WANT TO USE (find with nvidia-smi)
GPU_INDEX=0

# --- Workflow Steps ---

echo "======================================================"
echo " STEP 1: Starting the fine-tuning process... "
echo "======================================================"
echo "Container will run in the background. Script will wait for it to finish."
echo "To monitor progress in another terminal, run:"
echo "docker logs -f $CONTAINER_NAME"
echo ""

# Run the fine-tuning container in detached mode (-d)
docker run \
  -d \
  --name $CONTAINER_NAME \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 finetune_whisper.py

# Wait for the fine-tuning container to finish before proceeding
echo "Waiting for the fine-tuning to complete. This will take a long time..."
docker wait "$CONTAINER_NAME"
echo "Fine-tuning container has finished."


echo ""
echo "======================================================"
echo " STEP 2: Converting the fine-tuned model... "
echo "======================================================"

docker run --rm \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 convert_model.py


echo ""
echo "======================================================"
echo " STEP 3: Running transcription with the new model... "
echo "======================================================"

docker run --rm \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 transcribe.py

echo ""
echo "======================================================"
echo " Workflow complete! "
echo "======================================================"
