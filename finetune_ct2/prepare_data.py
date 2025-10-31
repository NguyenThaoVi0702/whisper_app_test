# prepare_data.py

import os
import json
import io
import logging
import threading

import boto3
import librosa
from datasets import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain
from transformers import WhisperProcessor

# --- Basic Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Paths ---
ANNOTATIONS_FILE_IN_CONTAINER = "/app/annotations.json"
BASE_MODEL_PATH = "./model"
PROCESSED_TRAIN_PATH = "/app/processed_data/train"
PROCESSED_VAL_PATH = "/app/processed_data/validation"

s3_client = None


def get_s3_client_for_worker():
    """Initializes and returns a boto3 S3 client for a single worker process."""
    try:
        client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION"),
        )
        return client
    except Exception as e:
        raise ConnectionError(f"Worker process failed to create S3 client: {e}")

# --- Data Loading and Parsing (Unchanged) ---
def load_and_parse_annotations(annotations_path):
    logging.info(f"Loading and parsing annotations from {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    data = {"audio_path": [], "transcription": []}
    skipped_count = 0
    for task in annotations:
        s3_uri = task.get("data", {}).get("audio")
        if not s3_uri:
            skipped_count += 1
            continue
        segments = []
        if task.get("annotations"):
            for result in task["annotations"][0].get("result", []):
                if result.get("from_name") == "transcription":
                    start_time = result.get("value", {}).get("start", 0)
                    text = " ".join(result.get("value", {}).get("text", [])).strip()
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
        segments.sort(key=lambda x: x[0])
        full_transcription = " ".join([text for start_time, text in segments])
        data["audio_path"].append(s3_uri)
        data["transcription"].append(full_transcription)
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} tasks due to missing audio URI or transcriptions.")
    logging.info(f"Successfully parsed {len(data['audio_path'])} valid data entries.")
    return Dataset.from_dict(data)

# --- Data Augmentation (Unchanged) ---
augment_pipeline = Compose([
    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
])

# --- Progress Logging and Processing Function ---
class ProgressTracker:
    def __init__(self, total, name=""):
        self.count = 0
        self.total = total
        self.name = name
        self.lock = threading.Lock()
    def increment(self):
        with self.lock:
            self.count += 1
        if self.count % 10 == 0 or self.count == self.total:
            logging.info(f"  --> [{self.name}] Processed {self.count}/{self.total} audio files...")

def process_and_tokenize_audio(batch, processor, tracker, is_training=False):
    """Downloads, processes, tokenizes audio, and updates progress."""
    global s3_client 
    if s3_client is None:
        s3_client = get_s3_client_for_worker()
        
    s3_uri = batch["audio_path"]
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        audio_bytes = response['Body'].read()
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        if is_training:
            audio_array = augment_pipeline(samples=audio_array, sample_rate=16000)
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        tracker.increment()
        return batch
    except Exception as e:
        logging.error(f"Failed to process {s3_uri}: {e}")
        return None

def main():
    if os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_VAL_PATH):
        logging.info("Pre-processed data already found on disk. Skipping data preparation.")
        return
        
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")
    raw_dataset = load_and_parse_annotations(ANNOTATIONS_FILE_IN_CONTAINER)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset_raw = dataset_dict["train"]
    val_dataset_raw = dataset_dict["test"]

    train_tracker = ProgressTracker(total=len(train_dataset_raw), name="Training")
    val_tracker = ProgressTracker(total=len(val_dataset_raw), name="Validation")

    logging.info(f"Processing {len(train_dataset_raw)} training samples with augmentation...")
    train_dataset = train_dataset_raw.map(
        lambda x: process_and_tokenize_audio(x, processor, train_tracker, is_training=True),
        num_proc=4, remove_columns=train_dataset_raw.column_names
    )

    logging.info(f"Processing {len(val_dataset_raw)} validation samples...")
    val_dataset = val_dataset_raw.map(
        lambda x: process_and_tokenize_audio(x, processor, val_tracker, is_training=False),
        num_proc=4, remove_columns=val_dataset_raw.column_names
    )
    
    logging.info("Saving processed datasets to disk for future runs...")
    train_dataset.save_to_disk(PROCESSED_TRAIN_PATH)
    val_dataset.save_to_disk(PROCESSED_VAL_PATH)
    logging.info("Data preparation complete.")

if __name__ == "__main__":
    main()
