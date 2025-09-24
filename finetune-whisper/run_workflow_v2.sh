#!/bin/bash

# --- Configuration ---
# Stop the script if any command fails.
set -e

# The name of your pre-built Docker image.
IMAGE_NAME="whisper-finetuner"

# A unique base name for the jobs to keep them organized.
JOB_BASENAME="whisper-job-$(date +%s)"

# The GPU index you want to use (find with nvidia-smi).
GPU_INDEX=0


# --- Workflow ---

echo "======================================================"
echo " LAUNCHING FULL WORKFLOW IN THE BACKGROUND "
echo "======================================================"
echo "Job Base Name: $JOB_BASENAME"
echo "Image: $IMAGE_NAME"
echo "GPU Index: $GPU_INDEX"
echo ""

# STEP 1: Run the fine-tuning container in the background.
FINETUNE_JOB_NAME="${JOB_BASENAME}-finetune"
echo "--> Step 1: Launching Fine-tuning (Name: $FINETUNE_JOB_NAME)"
docker run --rm -d \
  --name "$FINETUNE_JOB_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --workdir /app \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  python3 finetune_whisper.py

echo "    Fine-tuning is running in the background."
echo "    To monitor logs, run: docker logs -f $FINETUNE_JOB_NAME"
echo ""

# STEP 2: Run the model conversion, but only after fine-tuning is done.
# The '&' at the end of the block runs this entire section in the background.
{
  # This command will pause this part of the script until the container finishes.
  docker wait "$FINETUNE_JOB_NAME"

  CONVERT_JOB_NAME="${JOB_BASENAME}-convert"
  echo "--> Step 2: Fine-tuning complete. Launching Conversion (Name: $CONVERT_JOB_NAME)"
  docker run --rm -d \
    --name "$CONVERT_JOB_NAME" \
    --workdir /app \
    -v "$(pwd)":/app \
    "$IMAGE_NAME" \
    python3 convert_model.py
  
  echo "    Conversion is running in the background."
  echo "    To monitor logs, run: docker logs -f $CONVERT_JOB_NAME"
  echo ""

  # STEP 3: Run transcription, but only after conversion is done.
  docker wait "$CONVERT_JOB_NAME"

  TRANSCRIBE_JOB_NAME="${JOB_BASENAME}-transcribe"
  echo "--> Step 3: Conversion complete. Launching Transcription (Name: $TRANSCRIBE_JOB_NAME)"
  docker run --rm -d \
    --name "$TRANSCRIBE_JOB_NAME" \
    --gpus "\"device=$GPU_INDEX\"" \
    --workdir /app \
    -v "$(pwd)":/app \
    "$IMAGE_NAME" \
    python3 transcribe.py
  
  echo "    Transcription is running in the background."
  echo "    To monitor final output, run: docker logs -f $TRANSCRIBE_JOB_NAME"
  echo ""
  echo "======================================================"
  echo " All jobs launched. The full workflow will continue in the background. "
  echo "======================================================"

} & # The '&' runs the entire { ... } block in the background.

# The script will immediately print this and exit, leaving the chain reaction running.
echo "Script has finished launching all background tasks."
echo "You can now safely disconnect."
