#!/bin/bash

set -e

# --- Configuration ---
EXISTING_IMAGE_NAME="dso-nexus.vietinbank.vn/ai_docker/finetune_whisper_lora:v2"
MIG_DEVICE_UUID="MIG-****" 
FINETUNE_CONTAINER_NAME="lora-finetuning-job-debug" # Use a clear name

# --- Create the host cache directory ---
mkdir -p ./.cache

echo "======================================================"
echo " STEP 1a: Starting a persistent container for the job..."
echo "======================================================"

# The container now runs a command that never ends.
# Notice the command at the end is "sleep infinity" instead of "python3 ..."
docker run \
  -d \
  --name "$FINETUNE_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -e HF_HOME=/app/.cache \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=$MIG_DEVICE_UUID \
  --workdir /app \
  -v "$(pwd)":/app \
  -v "/tmp/viet_bud500":"/tmp/viet_bud500" \
  -v "$(pwd)/.cache:/app/.cache" \
  "$EXISTING_IMAGE_NAME" \
  sleep infinity # <-- THIS IS THE KEY CHANGE

echo "Container '$FINETUNE_CONTAINER_NAME' is running in the background."
echo ""
echo "======================================================"
echo " STEP 1b: Executing the fine-tuning script inside the container..."
echo "======================================================"

# Now, execute the script inside the container that's already running.
# We use "docker exec". The script's output will stream to your terminal.
docker exec -i "$FINETUNE_CONTAINER_NAME" python3 finetune_lora.py

# The script has now finished, either by succeeding or failing.
# THE CONTAINER IS STILL RUNNING.

echo ""
echo "======================================================"
echo " SCRIPT FINISHED. The container is still running."
echo "======================================================"
echo ""
echo "To debug, run the following command to get a shell inside the container:"
echo "  docker exec -it $FINETUNE_CONTAINER_NAME bash"
echo ""
echo "Once you are finished debugging, you must manually stop and remove the container:"
echo "  docker stop $FINETUNE_CONTAINER_NAME"
echo "  docker rm $FINETUNE_CONTAINER_NAME"
echo ""
