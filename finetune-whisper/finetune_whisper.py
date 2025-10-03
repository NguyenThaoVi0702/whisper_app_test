import os
import glob
from dataclasses import dataclass
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

# --- 1. Define Paths and Configuration ---
BASE_MODEL_PATH = "./model"
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"
DATASET_PATH = "/tmp/viet_bud500"
NEW_ADAPTER_SAVE_PATH = "./my-whisper-medium-lora-continued"

# --- 2. Load Model, Tokenizer, and Feature Extractor ---
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, use_cache=False, device_map="auto")

# --- 3. Load and Prepare PEFT Model for Continued Training ---
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)
model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))
model.print_trainable_parameters()

# --- 4. Load and Prepare the Dataset ---
parquet_dir = os.path.join(DATASET_PATH, "data")
data_files = {"train": glob.glob(f"{parquet_dir}/train-*.parquet")}
dataset = load_dataset('parquet', data_files=data_files).cast_column("audio", Audio(sampling_rate=16000))

def filter_data(example):
    return len(example["audio"]["array"]) > 16000 and len(example["transcription"]) > 5

print("Original dataset size:", len(dataset["train"]))
dataset = dataset.filter(filter_data, num_proc=1)
print("Filtered dataset size:", len(dataset["train"]))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)

# --- 5. Define Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding()

# --- 6. Define Training Arguments (with all fixes) ---
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=8,
    # --- CRITICAL FIX 3: Simulate a large batch size of 64 (8x8) ---
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    warmup_ratio=0.05,
    max_steps=1000,
    fp16=True,
    optim="adamw_bnb_8bit",
    do_eval=False,
    save_steps=200,
    save_total_limit=3,
    logging_steps=25,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    label_names=["labels"],
)

# --- 7. Create and Run the Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training with corrected hyperparameters, filtered data, and large batch simulation...")
trainer.train()

print(f"Training complete. Final adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
