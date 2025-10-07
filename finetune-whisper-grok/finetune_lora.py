# finetune_whisper.py (Quantization Disabled for A100 Stability)
import os
import glob
from dataclasses import dataclass
import torch
import numpy as np  # For potential array ops, but WER is pure Python here
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig

# --- Built-in Offline WER (No jiwer Needed) ---
def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings (word-level)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_wer(reference, hypothesis):
    """Compute Word Error Rate (WER) between lists of strings."""
    if len(reference) == 0 or len(hypothesis) == 0:
        return 1.0 if len(reference) != len(hypothesis) else 0.0  # Handle empty cases
    total_distance = sum(levenshtein_distance(ref.split(), hyp.split()) for ref, hyp in zip(reference, hypothesis))
    total_words = sum(len(ref.split()) for ref in reference)
    return total_distance / total_words if total_words > 0 else 0.0

# --- 1. Define Paths and Configuration ---
BASE_MODEL_PATH = "./model"  # Your coworker's pre-fine-tuned Whisper Large V3 Turbo
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"  # Existing adapter (renamed for clarity, but works with turbo)
DATASET_PATH = "/tmp/viet_bud500"  # Local path to vietbud500
NEW_ADAPTER_SAVE_PATH = "./my-whisper-medium-lora-continued"

# --- 2. Load Processor, Model, and Prepare for Continued PEFT Training ---
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

# DISABLED: Quantization (to avoid bnb/Triton issues on non-root Docker)
# quantization_config = BitsAndBytesConfig(...)  # <-- Commented out

model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    # quantization_config=quantization_config,  # <-- Disabled
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.bfloat16  # <-- A100-optimized precision (stable, no quantization needed)
)

# Critical: Reset decoder biases to prevent repetitive artifacts (e.g., "v v v v")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Prepare for k-bit training and load existing adapter for continued fine-tuning
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)

# Re-enable gradients for conv1 (Whisper-specific)
model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))

model.print_trainable_parameters()

# --- 3. Load and Prepare Dataset (Train + Val Splits Locally) ---
parquet_dir = os.path.join(DATASET_PATH, "data")
train_files = {"train": glob.glob(f"{parquet_dir}/train-*.parquet")}
val_files = {"train": glob.glob(f"{parquet_dir}/validation-*.parquet")}  # HF loads 'train' key for any split

# Load train and val
train_dataset = load_dataset('parquet', data_files=train_files).cast_column("audio", Audio(sampling_rate=16000))["train"]
val_dataset = load_dataset('parquet', data_files=val_files).cast_column("audio", Audio(sampling_rate=16000))["train"]

def filter_data(example):
    # Basic filter: >1s audio, >5 chars text, and <30s to avoid OOM
    audio_len = len(example["audio"]["array"])
    return 16000 < audio_len < 480000 and len(example["transcription"]) > 5

print("Original train size:", len(train_dataset))
train_dataset = train_dataset.filter(filter_data, num_proc=4)
print("Filtered train size:", len(train_dataset))

print("Original val size:", len(val_dataset))
val_dataset = val_dataset.filter(filter_data, num_proc=4)
print("Filtered val size:", len(val_dataset))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    # FIXED: Use .input_ids, not .ids
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    # Compute lengths for filtering
    batch["input_length"] = len(batch["input_features"])
    batch["labels_length"] = len(batch["labels"])
    return batch

# Map preparation
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=4)

# Filters for stability (from original script)
def filter_inputs(example):
    return 0 < example["input_length"] < 480000  # ~30s at 16kHz

def filter_labels(example):
    return example["labels_length"] < 448  # Max decoder length

train_dataset = train_dataset.filter(filter_inputs, input_columns=["input_length"])
train_dataset = train_dataset.filter(filter_labels, input_columns=["labels_length"])
val_dataset = val_dataset.filter(filter_inputs, input_columns=["input_length"])
val_dataset = val_dataset.filter(filter_labels, input_columns=["labels_length"])

# Clean up length columns
train_dataset = train_dataset.remove_columns(["input_length", "labels_length"])
val_dataset = val_dataset.remove_columns(["input_length", "labels_length"])

# Shuffle train
train_dataset = train_dataset.shuffle(seed=42)

# --- 4. Define Data Collator (Enhanced with Processor) ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor  # Pass processor for consistent padding

    def __call__(self, features):
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad using processor components
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens for loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS if added (Whisper appends it later)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Offline WER Metrics (Built-in, No External Libs)
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER (using built-in function)
    wer = compute_wer(label_str, pred_str)
    return {"wer": wer}

# --- 5. Training Arguments (Optimized for A100 with Eval; No bnb Optim) ---
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=16,  # A100 can handle larger
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch ~64
    learning_rate=1e-5,  # Slightly higher for continued tuning
    warmup_ratio=0.1,
    max_steps=500,  # Conservative for continued; monitor eval
    bf16=True,  # A100 native support
    optim="adamw_torch",  # <-- Changed: Default AdamW (no bnb_8bit needed)
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=3,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=False,  # PEFT compatibility
    dataloader_num_workers=0,  # <-- Temp: Disable multi-worker to avoid cache races (re-enable later if stable)
)

# --- 6. Create and Run Trainer with Eval ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=processor,  # FIXED: Use processor
    compute_metrics=compute_metrics,  # Offline WER tracking
)

print("Starting continued training with evaluation...")
trainer.train()

print(f"Training complete. Best adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
