Of course. This is an excellent next step for improving your fine-tuning pipeline. Using your direct labeling data is the best way to get high-quality results. Let's break this down into a complete, step-by-step guide.

Analysis of Your JSON Labeling File

First, I've analyzed the structure of your JSON file. It's a standard output format from a labeling tool like Label Studio. Here are the key takeaways:

The file is a list of tasks.

Each task contains data.audio, which holds the S3 URI for the audio file.

Each task has annotations[0].result, which is a list of labeled segments.

Within result, there are two types of segments we care about:

"from_name": "transcription": This contains the actual text in value.text[0].

"from_name": "label": This contains the speaker or noise label in value.labels[0].

Crucially, a single audio file can have multiple transcription segments from different speakers or be interrupted by segments labeled as "Noise".

Our strategy will be: For each audio file, we will find all transcription segments, filter out any labeled as "Noise", and concatenate the text in chronological order to form one complete transcript for that audio file.

Step 1: Securely Configure S3 Access in the Container

You must never hardcode credentials in your code. The best practice is to pass them to the Docker container as environment variables. We will also mount your CA certificate into the container.

Here is your updated run_temp.sh script. You will need to replace the placeholder values with your actual credentials.

Updated run_temp.sh
code
Bash
download
content_copy
expand_less
#!/bin/bash

set -e

EXISTING_IMAGE_NAME="finetune_whisper_lora:v3"
# Choose an available MIG device UUID from your nvidia-smi output
MIG_DEVICE_UUID="MIG-GPU-a45fc4b1-c85e-5a3b-8b3d-79191377ec06/4/0" # Example, verify this
FINETUNE_CONTAINER_NAME="lora-finetuning-job-s3"

# --- S3 Configuration (REPLACE WITH YOUR VALUES) ---
S3_ENDPOINT_URL="YOUR_S3_URL_ENDPOINT"          # e.g., "https://s3.your-company.com"
S3_ACCESS_KEY="YOUR_ACCESS_KEY"
S3_SECRET_KEY="YOUR_SECRET_KEY"
S3_REGION="YOUR_BUCKET_REGION"                  # e.g., "us-east-1"
S3_CA_BUNDLE_PATH="/path/to/your/ca-bundle.crt" # The path to your CA cert on the HOST machine

# --- Local Paths ---
# The local path to your JSON annotations file
ANNOTATIONS_FILE_PATH="$(pwd)/your_annotations.json"
# The name of the file inside the container
ANNOTATIONS_FILE_IN_CONTAINER="/app/annotations.json"

mkdir -p ./.cache

# Stop and remove the container if it already exists
docker stop "$FINETUNE_CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$FINETUNE_CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting fine-tuning container..."

docker run \
  -d \
  --name "$FINETUNE_CONTAINER_NAME" \
  --user $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --gpus "device=$MIG_DEVICE_UUID" \
  --workdir /app \
  \
  # --- Volume Mounts ---
  -v "$(pwd)":/app \
  -v "$(pwd)/.cache:/app/.cache" \
  # Mount the annotations file into the container
  -v "$ANNOTATIONS_FILE_PATH":"$ANNOTATIONS_FILE_IN_CONTAINER":ro \
  # Mount the CA certificate into the container
  -v "$S3_CA_BUNDLE_PATH":/app/ca-bundle.crt:ro \
  \
  # --- Environment Variables for S3 Access & Cache ---
  -e HF_HOME=/app/.cache/huggingface \
  -e TORCH_HOME=/app/.cache/torch \
  -e S3_ENDPOINT_URL="$S3_ENDPOINT_URL" \
  -e AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" \
  -e AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  -e AWS_REGION="$S3_REGION" \
  # Tell Python's boto3 library where to find the CA certificate
  -e AWS_CA_BUNDLE=/app/ca-bundle.crt \
  \
  "$EXISTING_IMAGE_NAME" \
  python3 finetune_lora.py

echo "Container '$FINETUNE_CONTAINER_NAME' started. Monitor logs with: docker logs -f $FINETUNE_CONTAINER_NAME"
Step 2: Update Your requirements.txt

Our new script will need libraries to interact with S3 and to perform audio augmentation. Add the following lines to your requirements.txt file and rebuild your Docker image if necessary.

code
Code
download
content_copy
expand_less
# Add these lines to your existing requirements.txt
boto3
audiomentations
Step 3: The New and Enhanced Fine-Tuning Script

This is the complete, final version of your finetune_lora.py. It replaces the old parquet loading logic with a robust system that loads data from your JSON file, fetches audio from S3, applies augmentations, and then fine-tunes the model.

finetune_lora.py (New Version for S3 Data and Augmentation)
code
Python
download
content_copy
expand_less
import os
import json
import io
import logging
from dataclasses import dataclass

import torch
import numpy as np
import boto3
import librosa
from datasets import Dataset, Audio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    utils,
)
from peft import PeftModel, prepare_model_for_kbit_training
import jiwer

# --- Basic Setup ---
utils.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Paths ---
BASE_MODEL_PATH = "./model"
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"
NEW_ADAPTER_SAVE_PATH = "./new_whisper_vietbud500_adapter_from_s3"
ANNOTATIONS_FILE_IN_CONTAINER = "/app/annotations.json"

# --- S3 Client Setup ---
def get_s3_client():
    """Initializes and returns a boto3 S3 client using credentials from environment variables."""
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION"),
        )
        # Verify connection by listing buckets
        s3_client.list_buckets()
        logging.info("S3 client initialized and connection verified successfully.")
        return s3_client
    except Exception as e:
        logging.error(f"Failed to create or verify S3 client: {e}")
        raise

s3_client = get_s3_client()

# --- Data Loading and Processing from JSON/S3 ---
def load_and_prepare_data_from_s3(annotations_path):
    """
    Parses the JSON annotation file, downloads audio from S3,
    and structures the data for the Hugging Face Dataset.
    """
    logging.info(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    data = {"audio_path": [], "transcription": []}
    skipped_count = 0

    for task in annotations:
        s3_uri = task.get("data", {}).get("audio")
        if not s3_uri:
            skipped_count += 1
            continue

        # Combine all transcription parts, ignoring "Noise" segments
        full_transcription = []
        # Gather all transcription segments with their start times
        segments = []
        if task.get("annotations"):
            for result in task["annotations"][0].get("result", []):
                if result.get("from_name") == "transcription":
                    start_time = result.get("value", {}).get("start", 0)
                    text = " ".join(result.get("value", {}).get("text", [])).strip()
                    # Check the corresponding label for this segment
                    label = "unknown"
                    for label_result in task["annotations"][0].get("result", []):
                        if label_result.get("id") == result.get("id") and label_result.get("from_name") == "label":
                           label = " ".join(label_result.get("value",{}).get("labels",[]))
                           break
                    if "noise" not in label.lower() and text:
                        segments.append((start_time, text))

        if not segments:
            skipped_count += 1
            continue

        # Sort segments by start time and join them
        segments.sort(key=lambda x: x[0])
        full_transcription = " ".join([text for start_time, text in segments])

        data["audio_path"].append(s3_uri)
        data["transcription"].append(full_transcription)

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} tasks due to missing audio URI or transcriptions.")

    logging.info(f"Successfully parsed {len(data['audio_path'])} valid data entries.")
    return Dataset.from_dict(data)

# --- Data Augmentation Pipeline ---
# This will be applied ONLY to the training data
augment_pipeline = Compose([
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.3),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
])

def download_and_process_audio(batch, is_training=False):
    """
    Downloads an audio file from S3, processes it, and optionally applies augmentation.
    """
    s3_uri = batch["audio_path"]
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        audio_bytes = response['Body'].read()
        
        # Load audio using librosa
        audio_array, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        # Apply augmentation only for the training set
        if is_training:
            audio_array = augment_pipeline(samples=audio_array, sample_rate=16000)

        batch["audio"] = {"path": s3_uri, "array": audio_array, "sampling_rate": 16000}
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    except Exception as e:
        logging.error(f"Failed to process {s3_uri}: {e}")
        # Return None or handle error appropriately to filter this example out
        return None


# --- MAIN SCRIPT EXECUTION ---

# 1. Load Processor
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

# 2. Model Loading and Hybrid Fine-Tuning Setup
use_quantization = False
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.bfloat16,
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)

logging.info("Unfreezing convolutional layers for hybrid fine-tuning...")
for name, param in model.model.model.encoder.named_parameters():
    if "conv" in name:
        param.requires_grad = True
model.print_trainable_parameters()

# 3. Load Data and Prepare Datasets
raw_dataset = load_and_prepare_data_from_s3(ANNOTATIONS_FILE_IN_CONTAINER)
# Split data into training and validation (90/10 split)
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)

logging.info("Processing training dataset with augmentation...")
train_dataset = dataset_dict["train"].map(lambda x: download_and_process_audio(x, is_training=True), num_proc=4)

logging.info("Processing validation dataset without augmentation...")
val_dataset = dataset_dict["test"].map(lambda x: download_and_process_audio(x, is_training=False), num_proc=4)

# Remove failed examples if any
train_dataset = train_dataset.filter(lambda example: example["audio"] is not None)
val_dataset = val_dataset.filter(lambda example: example["audio"] is not None)

# 4. Data Collator and Metrics (same as before)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# 5. Optimized Training Arguments
has_bf16 = torch.cuda.is_bf16_supported()
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=50,
    num_train_epochs=5, # Train for 5 epochs, good for small high-quality datasets
    bf16=has_bf16,
    fp16=not has_bf16,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False, # Important: Keep columns for processing
    label_names=["labels"],
    predict_with_generate=True,
    dataloader_num_workers=4,
    generation_max_length=448,
)

# 6. Initialize and Run Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processor=processor,
)

logging.info("Starting fine-tuning with data from S3 and audio augmentation...")
trainer.train()

logging.info("Final evaluation on the best model:")
final_metrics = trainer.evaluate()
logging.info(f"Final WER: {final_metrics['eval_wer']:.4f}")

logging.info(f"Training complete. Best adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
