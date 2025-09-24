#!/bin/bash
set -e

# --- Configuration ---
IMAGE_NAME="lora-finetuner" # You can keep the same image name
CONTAINER_NAME="full-finetune-job-v2" # Use a new name for the job
GPU_INDEX=0

# --- Build and Run ---
echo "--> Building the Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .

echo ""
echo "--> Starting the container '$CONTAINER_NAME' in the background..."
docker rm "$CONTAINER_NAME" 2>/dev/null || true

docker run \
  -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 full_finetune_from_merged.py # <-- THE ONLY CHANGE IS HERE

echo ""
echo "Container started successfully."
echo "---------------------------------------------------------"
echo "To monitor the training progress, run this command:"
echo "docker logs -f $CONTAINER_NAME"
echo "---------------------------------------------------------"
